# bitdelta_gemm：REG 现场解包 vs SHARED 展开复用的临界点

P2-P（Plan 2B）只回答一个问题：`bitdelta_gemm` 的 packed sign word 应该在**每个输出 tile 里现场解包**（REG，on-chip，无全局中间结果），还是**先展开成全局权重矩阵、屏障之后由 GEMM 读**（SHARED：producer → barrier → consumer）？本阶段在真实 Metal 上把每条合法形态的完整成本都测了，结论是 **保留 REG**，不实现 SHARED 候选。

`publication_allowed=false`。下面所有微秒数都是本阶段证据，不是生产 speedup 发布。

## 语义、字节与并行度

`tasks/bitdelta_gemm` 声明 `y[i,j] = scale · Σ_k x[i,k] · W[k,j]`，其中 `W[k,j] = +1` 当 `bits[k//32, j]` 的第 `k mod 32` 位为 1，否则 `-1`。`bits` 是 u32 `[KB, N]`（`KB = K/32`），所以 packed 体积是 `4·(K/32)·N = K·N/8` 字节，fp16 展开是 `2·K·N`，比值恒为 **16×**（fp32 展开 32×）。

| 档 | M | N | K | packed | 展开 fp16 | 展开 fp32 | 解码元素 | REG tile / grid / TG | SHARED tile / TG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `small` | 128 | 256 | 256 | 8 KiB | 128 KiB | 256 KiB | 65,536 | 32×32 / 8×4 / 32 | 64×64 / 8 |
| `paper` | 512 | 4096 | 4096 | 2 MiB | 32 MiB | 64 MiB | 16,777,216 | 64×64 / 64×8 / 512 | 64×64 / 512 |

解码复用次数由**该路线自己的**输出行块数决定，两条路线不能共用一个数：small 的 REG 是 32×32 tile（`grid=8,4,1`，四个行块），SHARED consumer 是 64×64（两个行块）；paper 两者都是 64×64。所以 REG 解码 4×K×N = 262,144（small）/ 8×K×N = 134,217,728（paper）个元素，而 SHARED 只解 `K·N` 个。同理 REG 的 packed 去重读是 32,768 / 16,777,216 B、发出 131,072 / 33,554,432 B（small 每个 word 被 4 个 lane 共用、paper 被 2 个共享），x 读取是 524,288 / 268,435,456 B。这些字段由候选源码的 launch grid、`TM/TN`（或 `BLOCK_M/BLOCK_N`）与解包 lane 映射独立推导，`_traffic()` 按路线分开计算；早期版本对 small 套用了 `M/64` 与 64 列几何，已作废（r4）。这笔 4×/8× 的解包节省是 SHARED 唯一可能的收益来源。

REG 读取的字节很少：每个 threadgroup 每个 k 步只读 32×64/32 = 64 个 packed word（paper 每两个 lane 共读一个 word，发出 512 B/步）。整个 paper 的 REG 发出 32 MiB packed 读取（去重后 16 MiB，全在 L2）。相反，SHARED 的 consumer 必须每行块读一次展开矩阵：paper 8×32 MiB = **256 MiB**，fp32 形态 8×64 MiB = 512 MiB。展开把“读 1/8 的字节 8 次”换成了“读 16 倍的字节 8 次”，只在解包 ALU 上省一次。

## REG 候选与硬件下限

候选树 [`tests/support/golden/metal/bitdelta_reg`](../../tests/support/golden/metal/bitdelta_reg)：

- `small`：2026-11-03 付费搜索的 init 候选（search best 2.765×），32×32 tile、4 个 SIMD group、每 4 个 lane 共用一个 packed word；唯一改动是 v2 的 `SCALE` attr 重绑（编译期常量，不是 buffer 槽），`y` 从 buffer(3) 移到 buffer(2)。SHA-256 `29fdc86f…`。
- `paper`：已登记 `gemm_paper_best.metal`，64×64×32 tile、4×4 个 `simdgroup_float8x8` 片段；改动只有 B 通路（u32 packed → `[n][k]` threadgroup tile + 转置 `simdgroup_load`）和 epilogue 里的 `SCALE` 乘。`load_a`/`store_a`/MMA 累加/`qid` 行号映射/输出 store 这五个片段与 fixture 逐字节相同（epilogue 不是逐字节相同）。SHA-256 `988317bd…`。

fresh（P2-I 公开 master key，搜索 seed 7）：

| 档 | fresh seed | REG | GPU 中位数 | cv | 硬件下限 | REG/下限 | 最快基线 | 延迟比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `small` | `4060777825911240` | **2.1044 µs** | 1.8223 µs | 0.0293 | 1.1362 µs | 1.852 | MLX-compile 12.7646 µs | 0.1649 |
| `paper` | `7898682953837593` | **1219.0597 µs** | 1208.8792 µs | 0.0029 | 1163.4844 µs | **1.048** | MLX-compile 1291.2111 µs | 0.9441 |

`paper` 的 roofline 是 `compute`（17.18 GFLOP / 实测 14.7659 TFLOPs = 1163.48 µs；最小字节 10.5 MB 只要 22.6 µs）。REG 已经贴在计算下限上，只高 **4.78%**。这决定了整件事的上限：

> **在同一个 fp16-MMA 算法内，任何“把解包搬出去”的改写最多只能回收 REG 高于硬件下限的那 55.58 µs（4.56%）。**

这不是外推，而是用同一份报告的 `roofline.lower_bound_us` 和 `candidate_us` 直接算出的比值；`decision.json` 里记作 `recoverable_decode_headroom_fraction=0.0456`。它**不**声称“换一个完全不同的算法也最多只能快 4.56%”——那是另一个问题，需要另开阶段；本阶段只比较 REG 解包与 SHARED 展开。

**判据是“比现在交付的 REG 更快”，不是“过登记的基线 1.10 门”。** 仓库的门是 `candidate_us / baseline_us ≤ 1.10`（`tcenv/portfolio_performance/performance.py::MAX_REGRESSION_RATIO`）。按本阶段的数字，paper 的最省 SHARED 估计 `1219.0597 × 1.0259 = 1250.59 µs` 对最快基线 MLX-compile 1291.21 µs 是 0.9685，**会通过**那个门；small 的 16.72 µs 对 12.76 µs 是 1.31，会失败。所以不能把 1.10 写成“SHARED 过不了”的理由——本阶段没有写 SHARED 候选，理由是它在两档都比 REG 慢（12/12 轮配对），以及 paper 的可回收空间只有个位数百分比。

`small` 是启动受限档（REG 是下限的 1.85 倍），硬件上还有余量，但任何两阶段路线都要多一个 stage；下面测到的两阶段下限本身就超过整个 REG。

六案不是只跑一遍：harness 的 correctness 对每个 case 跑三轮初值（out1 毒化、out2 零填充、out3 复跑毒化），要求三次输出逐字节相同、`unwritten=0`、`safety.violations=[]`。两档的 `nondeterminism` 都是空列表，`unwritten` 全为 0，说明结果既不依赖输出初值、也没有漏写元素，只读输入未被改写。`ieee` 的 NaN 逐位置与参考一致（`small` 768 个、`paper` 53288 个）。

## 合法 SHARED 形态只有三种

在这条工具链上 `uninitialized_poison` scratch 的第一个 producer 必须**声明且反射**为 `write`；非 const `device T*` 一律反射 `readwrite`（`u[tid]=1.0f` 也一样），只有 `texture2d<float, access::write>` 反射 `write`。因此：

1. **`shared_packed16`（本阶段新增，最省）**：poison fp32 纹理，一个 texel 装两个解码后的 half 位型（2 B/weight，与 fp16 展开同 footprint），consumer 每个 8 half 只做一次 16 B 拷贝，不需要 fill。这是合法形态里唯一同时避开 fill 与 fp32 双倍流量的路子。
2. **`shared_fp32`**：poison fp32 纹理，一个 texel 一个 fp32 值，consumer 转 half。不需要 fill，但 consumer 读 512 MiB（paper）。
3. **`shared_fp16`**：fp16 buffer scratch 必须声明 `device_zero`，harness 的 byte-granular fill 在**计时窗内**执行，每轮清零 `2·K·N` 字节（paper 32 MiB）。

第四条“标准 fp16 poison 展开”在合同里不存在；probe 只能通过把 `device_zero` 的 fill 从计时窗里拿掉来测它的下界（`no_fill` 臂，标注 `not_a_route`）。`host_written` fp16 scratch 需要候选 Host Program 在计时窗内从 CPU 写 16.7M 个 half，不可能快于 GPU producer 的 41–44 µs，本阶段不实现、也不当证据。

## probe：8 个臂 + 三档覆盖 + 同轮配对

`experiments/bitdelta_unpack_probe.py` 在真实 Metal 上测（`not_a_candidate=true`）：

| 臂 | 做什么 | small | paper |
| --- | --- | ---: | ---: |
| `floor` | 4 个 fp32 texel，producer → barrier → consumer，无 fill | 6.2565 µs | 2.2614 µs |
| `expand_fp32` | 合法 poison producer，逐元素写 `[K,N]` fp32 | 2.8090 µs | 220.9211 µs |
| `expand_fp16` | `device_zero` fp16 producer（fill + 写 `2·K·N`） | 19.9782 µs | 903.6974 µs |
| `fill_fp16` | 只测 harness 的 fill（+ 两个空核） | 14.8880 µs | 818.6364 µs |
| `packed` | **REG 候选源码，逐字节** | 1.9125 µs | 1245.2113 µs |
| `shared_packed16` | packed16 producer + GEMM consumer | 15.7252 µs | 1257.5208 µs |
| `shared_fp32` | fp32 producer + GEMM consumer | 16.3634 µs | 1356.1629 µs |
| `shared_fp16` | fp16 producer + fill + GEMM consumer | 29.3097 µs | 1813.3525 µs |
| `no_fill`（`shared_fp16`） | 同上，但把声明的 fill 拿出计时窗（不是路线） | 14.4144 µs | 1265.5083 µs |
| `no_fill`（`shared_packed16`） | 同上，packed16 本来无 fill | 15.3761 µs | 1253.9557 µs |

三个展开臂都不是自述字节数：producer 把每个 packed word 的精确 ±1 解码写进 scratch，运行后回读**整块** scratch，与 CPU 解码逐元素比较（fp16 比较 uint16 位型、fp32 比较 float、packed16 比较 half 位型）。三档的 `expected_elements = K·N`（16.7M at paper）全部 `mismatched_elements=0`。packed 与三个 shared 臂拿到同一份 bits、`x=1`，输出 `y` 必须逐元素相同（`packed_vs_shared_*_equal=true`）。

### probe 与 fresh 的口径差

probe 的 `packed` 臂就是候选源码逐字节，但它跑在原始批计时上（16 组轮转、每轮 60 ms 窗），没有 harness 的基线交错。因此两档的 probe/fresh 差不同：paper 是 +2.2%（1245.21 vs 1219.06），small 是 **−9.1%**（1.9125 vs 2.1044）——small 的每轮固定开销在 harness 协议里更大。这个差值是记录项不是门：决策用的是 fresh 的绝对成本、同一份报告算出的硬件下限，以及**同一次 probe 内**的配对比值（分子分母共享同一轮的口径）。

### 同轮配对 A/B

顺序测量不足以裁决几个百分点的问题：REG 臂在 r2/r3 两次运行之间漂了 6.7%，而 packed16 臂只漂 0.2%。所以 probe 增加了**同轮配对**：12 轮里每个臂背靠背各跑一次（各自轮转 16 个参数组），报告的是每轮比值的**中位数**，不是两个中位数的比值。

| 档 | `shared_packed16`/REG | `shared_fp32`/REG | `shared_fp16`/REG |
| --- | ---: | ---: | ---: |
| `small` | **7.9445**（7.69–8.28，12/12 慢） | 8.4109（8.13–8.79，12/12） | 15.2128（14.67–15.92，12/12） |
| `paper` | **1.0259**（1.0119–1.0374，12/12 慢） | 1.0918（1.0785–1.1129，12/12） | 1.4754（1.4547–1.5313，12/12） |

paper 上最省合法形态的**最好一轮**也慢 1.19%；最慢一轮慢 3.74%。没有任何一轮靠噪声反转。

归档重放（`evidence/unpack-probe-replay.json`，同机、另一次运行）复现了稳定比较：paper 的配对中位数 1.0255 / 1.0875 / 1.4739，排序与归档一致；small 的 8.2818 / 8.8025 与归档的 7.9445 / 8.4109 同量级。但 small 的**两个最省形态之间**不可靠：归档里 packed16 略优、评审的重跑里出现过 8.5589 对 8.5482 的翻转，归档重放的 small 绝对微秒也从 1.91 漂到 2.00。这正是用配对而不是两个绝对值的原因，也让 small 的结论只能读成“差一个数量级”，不能读成精确倍数或两者之间的排名。

## fill 的代价与“减法陷阱”

把单独测到的 `fill_fp16` 从 fp16 路线里减掉会得到 `1813.35 − 818.64 = 994.72 µs`，比同一 GEMM 的硬件下限 1163.48 µs 还低 13%，物理上不可能：byte-granular fill 是访存型、consumer 是计算型，跨轮转组时二者重叠，单独测 fill 得到的是“独占 GPU 的 fill 时间”，不是它在流水线里的边际成本。

实测的边际成本是 `1813.35 − 1265.51 = 547.84 µs`（占该路线 30.2%），而同一批 kernel 的 no-fill 流水线 1265.51 µs 与 `shared_packed16`（1257.52 µs，本来无 fill）在 0.6% 内互相印证。所以决策用**实测 no-fill**，不用减法外推；`decision.json` 把两种算法都记下来并给出 270.79 µs 的差值（`subtraction_error_us`），verifier 要求它为正。

## 决策

| 档 | REG | 最省合法 SHARED | 比值 | 对最快基线 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `small` | 2.1044 µs | 16.72 µs | 7.9445× | 1.31（会失败） | 排除 |
| `paper` | 1219.0597 µs | 1250.59 µs | 1.0259× | 0.9685（会通过） | 排除 |

`paper` 的行用配对比值折算：`1219.0597 × 1.0259 = 1250.59 µs`（`decision.json` 记录配对中位数与区间，probe 的顺序臂中位数是 1257.52 µs）。对基线 0.9685 说明它**不是**被基线门排除的；它被排除是因为比 REG 慢，且同一算法下没有两位数收益空间。

停的理由分三层，且互不依赖：

1. **结构排除 small**：两阶段下限在五次归档运行里是 2.68–7.47 µs，每次都高于 REG 2.1044 µs（当前 6.2565 µs = 2.973×；最小的一次 1.28×）；三个形态实测 7.9×–15.2×。
2. **上限约束 paper**：REG 已在硬件下限的 1.048×，可回收上限 4.56%，是个位数。
3. **实测排除三种合法形态**：最省形态 1.0259×、fp32 形态 1.0918×、fp16+fill 1.4754×，都是 12/12 轮更慢。

因此 `adopted_route=REG`、`decision=reg_only`、`shared_candidate_source_added=false`、`same_algorithm_ab_run=false`。没有为了“用上 ScratchSpec”而写一个已知更慢的候选。

## 与 conv2d、gemm_silu 的边界

| 阶段 | 中间结果 | 结论 | 判据 |
| --- | --- | --- | --- |
| P2-I（conv2d） | 全局 `U` 权重变换，被许多 tile 复用 | **SHARED 赢**（paper 2.39×） | producer 固定成本被大量 consumer 摊薄 |
| P2-O（gemm_silu） | 一次性激活中间结果 | REG | 省下的是几个 flop，任何全局往返都会输 |
| P2-P（bitdelta） | 展开后的权重矩阵 | **REG** | 省下的是 8× 解包，但换来 32 MiB 写 + 256 MiB 读 + 一个 stage；解包本身很便宜 |

不能把 conv2d 的正例套到 bitdelta：conv2d 的 producer 做的是**大量算术**（Winograd 变换）并被 36 个位置复用；bitdelta 的 producer 只是位运算 + 写，而 consumer 本来就读得很少。也不能把这里的负例推广成“全局 scratch 永远没价值”：判据是“被摊薄的重复计算”与“新增流量 + stage”的大小关系。

## 证据、重放与边界

- fresh subject：[`reg-small-confirmation.json`](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/reg-small-confirmation.json)、[`reg-paper-confirmation.json`](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/reg-paper-confirmation.json)。
- probe：[`unpack-probe.json`](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe.json)，以及被取代的 r1（B staging 越界）/r2/r3 原字节。
- 决策：[`decision.json`](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/decision.json)，由记录脚本从 fresh + probe 生成，verifier 逐字段重算。
- 验收记录：[`P.md`](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/P.md)。

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/record.py --decision
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/verify.py
```

边界：`publication_allowed=false`；没有 `taic` 付费搜索、没有模型调用；没有改提示面/`current_prompt_sha1`、六 RPC、字段号、HMAC、BlobChannel；没有重签 trust；NPU 仍 simulation-only。结论只覆盖登记的两个 fp16 形状，不外推到其它 M/N/K、bf16 或其它 GPU。

## 剩余限制

- paper 的最省合法形态只慢 2.59%，是“可信地不赢”，不是数量级差距；真正的分界来自硬件下限（REG 1.048× 下限）。
- `shared_packed16` 依赖“R32Float 纹理写逐位保真”这一实测性质，两档都由整块回读验证，但没有在其它 Apple GPU 上验证。
- probe 的 shared 臂统一用 64×64 tile；small 的 REG 是 32×32（两条路线的块数在 evidence 里分开记录）。small 的结论不依赖这一点：两阶段下限在五次归档运行里是 2.68–7.47 µs（1.28×–3.55× REG，当前 6.2565 µs = 2.973×），已经单独高于 REG；三个合法形态实测才是 7.9× 以上。不要拿 small 的 shared 臂绝对值去推断“换 tiling 就能追平”。
- 没测 `host_written` fp16 scratch（需要候选 Host Program，且在计时窗内从 CPU 写 16.7M 个 half）。
- 本阶段的完整快速 CPU 套件（`-m 'not gpu and not llm and not slow'`）是 **5185 passed、13 skipped、32 failed**；32 个失败全部落在 selfcheck/harness-overlay/进程权限这一类既有失败上，与 P2-P 无关：其中 18 个是 `AF_UNIX path too long`（本会话的 TMPDIR 过长，换短 TMPDIR 后 `test_selfcheck_bridge.py` 20/20 通过），其余 14 个在 `git archive HEAD` 的**干净检出**上 1:1 复现（`ps` 权限、进程组、selfcheck host record、harness overlay/patch）。P2-P 自己的 8 个 CPU 节点与 5 个 Metal 节点全绿，不能把整张报表写成“全绿”。
