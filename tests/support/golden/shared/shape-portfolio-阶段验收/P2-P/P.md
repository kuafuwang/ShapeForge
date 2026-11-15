# P2-P：bitdelta_gemm REG 现场解包 vs SHARED 展开复用

- epoch：`p2-p-bitdelta-reg-vs-shared-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`；NPU 仍 simulation-only

本阶段只做 Plan 2B。P2-I/J/K/L/M 已提交，历史未改；P2-N/O 的未提交工作树原字节保留，本阶段没有修改它们引用的任何文件，也没有 `git commit/reset/clean`。

## 结论

`bitdelta_gemm` 的 packed sign 应当在**每个输出 tile 现场解包（REG）**，不把展开后的权重矩阵写进全局 scratch。两档登记形状的全部合法 SHARED 形态在同轮配对里都更慢（paper 1.0259×/1.0918×/1.4754×，small 7.94×/8.41×/15.21×），paper 的可回收上限也只有 4.56%（同一算法内），因此**没有实现 SHARED 候选**，也没有为了使用 ScratchSpec 写一个已知更慢的核。

| 档 | REG fresh | 硬件下限 | REG/下限 | 最省合法 SHARED（同轮配对） | fp32 poison SHARED | fp16 + device_zero fill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | **2.1044 µs** | 1.1362 µs | 1.852 | **7.9445×** REG | 8.4109× | 15.2128× |
| `paper` | **1219.0597 µs** | 1163.4844 µs | 1.048 | **1.0259×** REG | 1.0918× | 1.4754× |

“最省合法 SHARED”是把两个解码后的 half 位型打进一个 fp32 texel 的 `uninitialized_poison` 纹理 producer：不需要 fill、展开 footprint 与 fp16 相同（2 B/weight），consumer 每 8 个 half 只做一次 16 B 拷贝。它在 paper 慢 2.59%，12/12 轮配对全部更慢（每轮比值 1.0119–1.0374，全部 > 1）。在 `small` 上它与 fp32 形态不可分辨（当前 7.9445 对 8.4109，归档重放 8.2818 对 8.8025，评审重跑曾出现 8.5589 对 8.5482 的翻转），所以 small 只声明“都慢一个数量级”，不声明两者之间的排序。

`paper` 是计算受限档：同一 GEMM 的硬件下限 1163.48 µs，REG 只比它高 4.78%（55.58 µs = 4.56%）。也就是说，**在同一个 fp16-MMA 算法内，即使把 REG 的解包开销全部免费消除**，收益上限也只有 4.56%（个位数），而展开路线不但不能免费，还要多一个 producer、一次 barrier、一次展开写和每行块一次展开读。

判断不是“能不能过登记基线的 1.10 门”，而是“能不能比现在交付的 REG 更快”。登记基线的 1.10 门是 `candidate_us / baseline_us ≤ 1.10`（`tcenv/portfolio_performance/performance.py::MAX_REGRESSION_RATIO`）。按本阶段数字，paper 的最省 SHARED 估计是 `1219.0597 × 1.0259 = 1250.59 µs`，相对最快基线 MLX-compile 1291.21 µs 是 **0.9685**，**它其实会通过那个门**；small 的估计 16.72 µs 相对 12.76 µs 是 1.31，会失败。所以本阶段的判据是“实测比 REG 慢”，不是基线门；把 1.10 写成“SHARED 过不了”的口径是错误的，已经删除。

## REG 候选与 fresh

候选树 [`tests/support/golden/metal/bitdelta_reg`](../../../../../../tests/support/golden/metal/bitdelta_reg) 两份都是单 stage、零 scratch、tile 内位解包：

| 档 | 来源 | tile | launch | SHA-256 |
| --- | --- | --- | --- | --- |
| `small` | `runs/apple-m4-max/20261103-090118_bitdelta_gemm_fp16_deepseek-flash_hh8_ynro/candidates/000_init/kernel.metal`（2026-11-03 搜索 init，search best 2.765×），只做 v2 的 `SCALE` attr 重绑 | 32×32 | `grid=8,4,1 threads=128,1,1` | `29fdc86fe1af731173b3311d6102953eabc77a86e0f0effadcbb9c03455c2697` |
| `paper` | `tests/support/golden/metal/fixtures/gemm_paper_best.metal`，只改 B 通路（u32 packed word → `[n][k]` threadgroup tile）与 epilogue 的 `SCALE` 乘 | 64×64×32 | `grid=64,8,1 threads=128,1,1` | `988317bdccaa052aa348a820f2c5490a4a13873adaf789082b3452ccb8a504f7` |

`paper` 的同源性不是注释声明：`assert_source_provenance()` 要求 `load_a`/`store_a`/MMA 累加/`qid` 行号映射/输出 store 这五个片段与已登记 fixture **逐字节相同**。改动只有两处：B 通路（packed word → `[n][k]` tile）和 epilogue 里的 `SCALE` 乘（fixture 没有这个 attr）。

两档 fresh 都用 P2-I 的公开 master key 派生 fresh seed（搜索 seed 仍是 7），identity `match`，第一次确认即 `confirmed`：

| 档 | fresh seed | REG fresh | 最快基线 | 延迟比 | 相对最快基线 | 次快基线 | 相对次快 | cv |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `small` | `4060777825911240` | **2.1044 µs** | MLX-compile 12.7646 µs | 0.1649 | 6.066× | torch-compile 30.8148 µs | 14.643× | 0.0293 |
| `paper` | `7898682953837593` | **1219.0597 µs** | MLX-compile 1291.2111 µs | 0.9441 | 1.0592× | torch-compile 1368.8646 µs | 1.1229× | 0.0029 |

六案（`random`、`edge_bits`、`large`、`tiny`、`outlier`、`ieee`）全部通过，每个 case 都跑了三轮初值（out1 毒化、out2 零填充、out3 复跑毒化），`small` 最大 scaled error 是 `tiny` 的 0.6008，`paper` 是 `tiny` 的 0.2781，门限都是 1.0。`ieee` 的 NaN 逐位置与参考一致（`small` 768 个、`paper` 53288 个）。`safety.ok=true`、`nondeterminism=[]`，说明结果不依赖 out1/out3 的毒化初值和 out2 的零初值，只读输入未被改写。两档都过 1.10 延迟门。

## 语义与形状代数

`bits` 是 u32 `[KB,N]`：`bits[k//32, j]` 的第 `k mod 32` 位是 `W[k,j]` 的符号（1 → +1，0 → −1）。因此：

| 量 | `small` | `paper` |
| --- | ---: | ---: |
| packed bytes `4·(K/32)·N` | 8,192 B | 2,097,152 B（2 MiB） |
| 展开 fp16 `2·K·N` | 131,072 B（16×） | 33,554,432 B（32 MiB，16×） |
| 展开 fp32 `4·K·N` | 262,144 B（32×） | 67,108,864 B（32×） |
| 解码元素 `K·N` | 65,536 | 16,777,216 |
| **REG tile / grid** | **32×32 / 8×4** | **64×64 / 64×8** |
| REG 行块 / 列块 / 输出 TG | **4 / 8 / 32** | 8 / 64 / 512 |
| REG 实际解码元素 | **262,144（4×）** | 134,217,728（8×） |
| REG packed 读取（去重/发出） | **32,768 / 131,072 B** | 16,777,216 / 33,554,432 B |
| REG x 读取 | **524,288 B** | 268,435,456 B |
| **SHARED tile / 行块 / 列块 / TG** | 64×64 / 2 / 4 / 8 | 64×64 / 8 / 64 / 512 |
| SHARED consumer 读展开矩阵 | 262,144 B | 268,435,456 B（256 MiB） |

REG 的解包重复次数等于**它自己的**输出行块数（每个行块都要读同一份 packed word），SHARED 只解一次。两条路线的 tile 不同，块数必须分开算：small 的 REG 是 32×32（4 个行块、32 个 TG），SHARED consumer 是 64×64（2 个行块、8 个 TG）；paper 两者都是 64×64。早期版本用 `M/64` 套在 small 的 REG 上，把 4× 解包误记成 2×、把 32,768 B 去重读误记成 16,384 B、把 524,288 B 的 x 读取误记成 262,144 B，并把 64 列几何套给了 8 列块；这些字段现在由候选源码的 launch grid 与 tile 常量独立推导，probe、decision 和测试都按推导值断言。

这正是要回答的问题：**把 4×（small）/8×（paper）重复解包换成 1× 解包 + 展开写 + 每行块一次展开读 + 一个额外 stage，是否划算？** 实测不划算。

## SHARED 的三种合法形态与理论无 fill 下界

在这条工具链上，`uninitialized_poison` scratch 的第一个 producer 必须声明的和反射的 access 都是 `write`；非 const `device T*` 一律反射 `readwrite`，只有 `texture2d<float, access::write>` 反射 `write`，因此 poison producer **只能是 fp32 纹理**。可选形态如下：

| 形态 | 合法性 | 是否需要 fill | 展开 footprint | paper 配对比值 |
| --- | --- | --- | --- | ---: |
| `shared_packed16`：poison fp32 纹理，一个 texel 装两个解码后的 half 位型 | 合法 | 否 | 2 B/weight | **1.0259** |
| `shared_fp32`：poison fp32 纹理，一个 texel 一个 fp32，consumer 转 half | 合法 | 否 | 4 B/weight | 1.0918 |
| `shared_fp16`：fp16 buffer scratch + `device_zero` | 合法 | 是（K·N halves） | 2 B/weight | 1.4754 |
| `shared_fp16`（去掉计时窗内的 fill） | **不是路线**（`device_zero` 必须每轮清零，probe 故意跳过） | 否 | — | 1.0163（1265.51/1245.21，独立测量） |

`shared_packed16` 是本阶段新写的探针臂，也是这次的关键证据：它证明“fp16 质量的无 fill 展开”本身并不能赢，1.0259× 与 fill 无关。纹理写是否逐位保住 half 位型（大量 NaN/亚正规位型）由整块 scratch 回读逐元素验证：三档 `mismatched_elements=0`。

**fill 的边际代价不是单独的 fill 臂。** 把单独测到的 `fill_fp16`（paper 818.64 µs）从 fp16 路线里减掉会得到 994.72 µs，**低于同一 GEMM 的硬件下限 1163.48 µs**，物理上不可能。原因是 byte-granular fill 是访存型、consumer 是计算型，跨轮转组时二者重叠。同一批 kernel 实测的 no-fill 流水线是 1265.51 µs，比减法结果高 270.79 µs；所以本阶段用**实测 no-fill**当无 fill 下界，不用“最差 fill 臂做减法”的外推。

## 为什么没有做 SHARED A/B

三条停止条件都触发，且结论不依赖某一次最差样本：

1. **small 被结构排除**：两阶段下限在五次归档运行里是 2.68–7.47 µs，每次都远高于 REG 2.1044 µs（当前这次 6.2565 µs = 2.973×；最小的一次也有 1.28×）；三个合法 SHARED 形态实测 7.9×–15.2×。small 是启动受限档，展开一次 131 KB 也要多一个 stage。
2. **paper 的收益上限是个位数**：REG 只比硬件下限高 4.56%，在同一个 fp16-MMA 算法内任何路线的最好可能收益都不超过这个数；即使把 REG 的解包全部免费消除也拿不到两位数。
3. **最省合法形态实测更慢**：同轮配对的 12/12 轮全部 > 1，中位数 1.0259（最小轮 1.0119）。展开路线没有省掉任何 FLOP，却多了一个 producer、一个 barrier、一次展开写和每行块一次的展开读。

这三条都不依赖“基线 1.10 门”（paper 的 SHARED 估计反而会通过它）。

因此 `implementation.shared_candidate_source_added=false`、`same_algorithm_ab_run=false`。这是 **negative result**：SHARED 被实测否决，REG 保留。不是因为预算或模型费用，也不是为了省事。

## probe 的 r1 缺陷与重录链

第一版 probe（actor 断网前录的 `unpack-probe.json`）有一个真实缺陷：所有 GEMM 臂的 B staging 每线程做 **4** 遍 `uint4`，把 64 个 k 行写进只有 `BLOCK_K=32` 行的 threadgroup 区域——越过声明数组 2296 halves（4592 B），并且每轮 k 步把展开矩阵读了两遍。这会把两个 shared 臂都抬高。

处理方式（原字节全部保留）：

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| `evidence/unpack-probe-r1-superseded.json` | r1 原始字节，schema `…probe.1` | 保留，不进入决策 |
| `evidence/unpack-probe-r2-superseded.json` | r2：staging 修好（pass 数由 `BLOCK_K·BLOCK_N` 推导） | 保留 |
| `evidence/unpack-probe-r3-superseded.json` | r3：加入 `shared_packed16` 与实测 no-fill | 保留 |
| `evidence/unpack-probe-r4-superseded.json` | r4：加入同轮配对 A/B，但 REG 几何按 `M/64` 记错（small 记成 2 个行块） | 保留，不进入决策 |
| `evidence/unpack-probe-replay-r4-superseded.json` | r4 那版重放的包装文档（内部仍是旧几何） | 保留 |
| `evidence/unpack-probe.json` | r5：按候选源码独立推导 REG 几何（small 4 行块 / 32 TG），修复账目；实测臂与配对方法不变 | 当前证据 |

当前 r5 的 `b_staging_audit` 逐档记录 `rows_written=32/rows_declared=32`、`over_read_ratio=1.0`、`threadgroup_overrun_bytes=0`，并把 r1 的缺陷常数（2.0×、4592 B）一起记进同一文档；`assert_probe()` 要求这两组值同时成立，旧版本不可能通过。缺陷的影响是实测的：r1→r2 让 paper 的 `shared_fp32` 从 1390.14 µs 降到 1343.26 µs（−3.4%），而 fill 占主导的 `shared_fp16` 几乎不动（1825.20 → 1828.41，+0.2%）。r2 只有两臂（packed16 是 r3 才加的）：`shared_fp32` 在 r2/r5 是 17.3319/16.3634 µs（small）与 1343.26/1356.16 µs（paper），`shared_fp16` 是 29.8861/29.3097 µs（small）与 1828.41/1813.35 µs（paper），差异都在同量级噪声内。结论方向不变。

## 证据与验收

`evidence/`：

- `reg-small-confirmation.json`、`reg-paper-confirmation.json`：完整 fresh subject（含 search/confirmation 两份原始报告）；
- `unpack-probe.json`：8 个臂 + 三档解码覆盖 + 同轮配对 A/B + 实测 no-fill，`not_a_candidate=true`；
- `unpack-probe-replay.json`：同机另一次运行的包装文档，只作旁证（paper 排序复现、small 两个最省形态不可分辨）；它**不是**决策输入，`decision_document()` 从不读它；
- `unpack-probe-r1/-r2/-r3-superseded.json`：缺陷与两次中间版本原字节；
- `unpack-probe-r4-superseded.json`、`unpack-probe-replay-r4-superseded.json`：几何记错的 r4 探针与其重放包装，保留但不进入决策；
- `decision.json`：由 fresh + probe 重算的决策，verifier 逐字段重算。

重录与重放：

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/record.py            # fresh + probe + decision
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/record.py --decision  # 只重算决策
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/verify.py             # 8 CPU + 5 真实 Metal
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/verify.py --static    # 只跑 CPU 节点
```

单个节点：

```bash
uv run pytest -q -n 0 tests/shared/test_bitdelta_unpack_decision.py
uv run pytest -q -n 0 -m gpu tests/backends/metal/test_bitdelta_unpack_gpu.py
```

性能与结论的完整推导见 [bitdelta-unpack-headroom.md](../../../../../../docs/架构/bitdelta-unpack-headroom.md)。

## 边界

- `publication_allowed=false`：`small` 的 6.066× 与 `paper` 的 1.0592× 只是本协议的相对位置，不是生产 speedup 发布。
- 没有启动 `taic` 付费搜索、没有模型调用；没有改提示面/`current_prompt_sha1`、六 RPC、字段号、HMAC、BlobChannel；没有重签 trust。
- 清微 NPU 仍只有模拟器，本阶段没有跑真实 NPU，也没有改 `tests/support/npu_sim/`。
- P2-I/J/K/L/M/O 的 manifest 逐文件不变；P2-N/O 未提交工作树原字节保留。任务包闭包仍是 79 个文件，摘要 `5512395852…`。
- 结论只覆盖登记的两个形状与 fp16；不外推到其它 M/N/K、bf16、其它 GPU（例如带宽/算力比不同的桌面 GPU），也不外推到“全局 scratch 永远没价值”。

## 剩余限制

- `probe` 的 shared 臂用 64×64 tile，`small` 的 REG 是 32×32（两条路线的块数在 evidence 里分开记录）；small 的结论不依赖 shared 臂的 tiling：两阶段下限单独就高于 REG，且三个形态实测都是 7.9× 以上。
- paper 最省合法形态只慢 2.59%，属于“可信地不赢”而不是“差一个数量级”。真正的分界来自硬件下限：REG 已在 1.048× 下限，展开路线的额外 stage 与流量没有可回收的余量。
- `shared_packed16` 依赖“fp32 纹理写逐位保真”这一实测性质；它在两档都由整块回读验证，但没有在其它 Apple GPU 上验证。
- `tiny` case 在 small 上是 0.6008、paper 0.2781，离 1.0 有余量但不是很大；改解包位型或改 `SCALE` 处理需要重新检查这一档。
- 没有测 `host_written` fp16 scratch（Host Program 在计时窗内写 16.7M 个 half）。它在结构上不会更快：同一份展开数据的 CPU 写不可能快于 GPU producer 的 41–44 µs，而且需要候选 Host Program bundle，超出本阶段范围。
- 独立评审共走了 r1–r8 轮，全部记录在 `review.json`（`prior_rounds`）与 `review-r4-superseded.json` 里：r1–r3 被拒绝；r4 接受后，人类复核推翻了它并指出两处硬伤——small 的 REG 几何被按 `M/64` 记错、以及把“比 REG 快 10%”误当成登记的基线 1.10 门；r5 重录后 r5–r7 又逐条修了残留数字/口径；r8 接受，r9 只复核这处文字、不改证据。当前验收节点只断言“每个合法形态都比 REG 慢”和 paper 的稳定排序，small 只断言数量级与彼此差 <25%；归档重放（`unpack-probe-replay.json`）把这一点固化。
