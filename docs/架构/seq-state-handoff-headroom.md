# 因果状态跨 stage 传递（SEQ）的代价与边界

这份文档回答 Plan 2C 的问题：**什么时候有因果顺序的状态必须跨 stage 传递，现有协议是否安全，这笔 handoff 写读加调度成本是否真的值得付。** 结论先行：现有 SEQ 协议安全，回写与状态的检查点足够；在允许的同步模型下（串行 encoder + buffer barrier）切链**不会增加并发**，所以它的成本是纯支出，`handoff_states` 继续留在 OpSpec workspace，**本阶段不新增 ScratchSpec**，也不改任何 ABI。

证据在 `tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/`。所有数字都是本机 Apple M4 Max 上的实测，`publication_allowed=false`，不是 speedup 发布。

## 1. 三种"状态"要分开

先分清三件事，否则会把 conv2d 的 `U` 复用、varlen 的 work list 和这里的因果状态混成一件事。

| 概念 | 谁拥有 | 在哪里 | 生命周期 | 能否跨 stage 复用 |
| --- | --- | --- | --- | --- |
| OpSpec workspace | 任务合同 | 命名张量，容量进 ABI | 每次调用有效 span，正确性 out1/out3 毒化、out2 清零 | 可以，读写权限由 stage 反射检查 |
| 候选 ScratchSpec | 候选几何 | 候选声明，容量进几何身份 | 每个计时迭代由 producer 重写（poison）、计时内清零（zero）或 Host 写（host_written） | 可以，但受初值策略约束 |
| 因果递推状态 | 候选的算法内部 | 递推 kernel 的线程/threadgroup 内存 | 一个 chain 内由因果顺序决定 | 只有被切开时才需要外化 |

`tasks/kda` 与 `tasks/varlen_kda` 把外化后的状态声明成 OpSpec workspace：

| 张量 | 作用 | 容量公式 |
| --- | --- | --- |
| `work_items` | 候选工作项表，位布局由候选决定 | `kda` 固定 `3×2` u32；`varlen_kda` `num_seqs*H*3×2` u32 |
| `item_counts` | 本次实际发布的工作项数 | 1 个 i32（`kda`）或 `num_seqs*H*3` 个（`varlen_kda`） |
| `handoff_states` | piece 边界上的 fp32 状态 | `kda` `2*H*D*D*4` B；`varlen_kda` `num_seqs*H*2*D*D*4` B |
| `handoff_flags` | ready/consumed 协议（只有 `varlen_kda` 有） | `num_seqs*H*4` i32 |

`tasks/mamba2_chunk_scan` 是第三种情况，值得对照：它的 chunk 状态 `states[BATCH,NCHUNKS,NHEADS,HEADDIM,DSTATE]` 是**合同输入**，chunk 内的递推由 `dA_cumsum` 和 `states` 一起并行求出。也就是说这份任务把"跨 chunk 状态"整体交给调用方，kernel 内部不需要 SEQ。它的 OpSpec 没有任何 workspace。**能做并行扫描的任务不需要状态交接；需要交接的只有严格逐 token 的递推。**

## 2. 现有 SEQ 路径怎么走

固定长度 `tasks/kda` 的候选是 `tests/support/kda_fixed_bundle.py` 打包的 msl-bundle：

```text
Host Program (host/tiling.c, native64)
  pieces = min(3, N)；均衡切分，每 piece 非空
  只发布 work_items 与 item_counts；不写 device-owned 的 handoff_states
        |
        v
stage 0  kda_piece0  grid=H  threads=1
  从 initial_state 读状态；处理 work_items[0]
  若无后续 piece 则写 final_state，否则写 handoff_states[0*H*D*D + ...]
        |  serial encoder + memoryBarrierWithScope(MTLBarrierScopeBuffers)
stage 1  kda_piece1  grid=H  threads=1
  从 handoff_states[(1-1)*H*D*D + ...] 读状态
  published < 2 时直接返回（Dyn-0：静态 grid，运行时 count）
        |  同上 barrier
stage 2  kda_piece2  grid=H  threads=1
  从 handoff_states[...] 读状态；写 final_state
```

`varlen_kda` 的 `piece_handoff.metal` 是同构的更完整版本：plan stage 读**本次实际** `cu_seqlens`、按 `min(3,len)` 均衡切分、把 flag 复位成 `FLAG_EMPTY`、给空序列materialize `final_state = initial_state`；三个 piece stage 用 `handoff_flags` 的 ready/consumed 协议交接，末 piece 还要求本链所有 flag 已被清空。

两条路径都**不使用跨 threadgroup 自旋**。stage 之间的顺序来自同一命令缓冲里的串行 encoder 与 buffer barrier；这是 Apple GPU 上没有可移植跨 TG 前向进度承诺时的唯一安全做法。

### 2.1 为什么"切开"在这套交付里不产生并发

KDA 上游之所以切链，是因为它把不同的 piece 分给**不同的 CTA 同时**跑，再用 acquire/release flag 交接。我们的 Metal 交付里 piece 落在**不同的 stage**，而 stage 之间是 barrier 定序，所以：

- 同一个 chain 的三个 piece 永远不重叠执行；
- 切链把一条 chain 的总 token 工作量拆成三段，但段间有两次完整的状态外化与重新加载；
- 只有当一条 chain 长度超出单个 threadgroup 的预算，或者 grid 远小于核数需要加 threadgroup 时，切链才可能换来并行度。登记形状（`kda` H=2/8，`varlen_kda` mixed 6 条链）都不属于这两种情况。

要真正拿到并发，就得让同一条 chain 的不同 piece 在**同一 stage** 里并行并互相等待 flag —— 那正是本项目明确不接受的做法（死锁风险，且没有可移植保证）。所以这一节的结论不是"实现得不好"，而是"在允许的同步模型下没有可回收的并发"。

## 3. 实测：切链到底花多少钱

### 3.1 同算法三臂（harness 端到端，搜索种子 7，各 3 次独立 full run）

三臂的 device 源码**逐字节相同**，只改 Host Program 的 piece 数和候选自己几何里的 stage 数：

| 档 | A：三 piece / 三 stage（现有） | B：单 piece / 三 stage | C：单 piece / 单 stage | A 相对 C | 其中 handoff (A−B) | 其中 stage floor (B−C) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kda single`（N=1） | 153.81 µs | 152.54 µs | **137.65 µs** | +11.7% | +1.27 µs | +14.89 µs |
| `kda small`（N=14） | 247.44 µs | 221.40 µs | **201.13 µs** | +23.0% | +26.05 µs | +20.26 µs |
| `kda tail`（N=17） | 253.01 µs | 227.80 µs | **212.37 µs** | +19.1% | +25.20 µs | +15.44 µs |
| `kda paper`（N=256, H=8, D=64） | 188402.36 µs | 187476.21 µs | **187375.16 µs** | +0.5% | +926.16 µs | +101.05 µs |
| `varlen mixed`（3 段 4/8/2） | 53.72 µs | 34.35 µs（单 piece，仍是 4 个 stage） | — | +56.4% | +19.36 µs | 同 stage 数，无此差 |

`kda single` 的 N=1 只有 1 个 piece，A 与 B 是同一份 work list（153.81 vs 152.54 µs，差 0.8%），这本身是一个自检：两条臂的差别必须消失才说明差异来自切链而不是别的。

`varlen mixed` 的两条臂都保留 plan + 三个 piece stage 的 4 次 dispatch，所以 19.36 µs 全部是 handoff 与 flag 协议的成本，与 stage floor 无关。

### 3.2 拆开看：state 拷贝、barrier、stage 调度

`experiments/seq_handoff_probe.py` 用同一份 `host_piece_handoff.metal` 和一个空阶段流水线单独量这几项（`not_a_candidate=true`）。GPU 时间是同一条命令缓冲里 `iters=100` 的中位数。

| 项 | 本次归档实测 | 说明 |
| --- | ---: | --- |
| 多一个空 stage（1→4 stage 斜率） | −0.2 µs（不可分辨） | 8 个 threadgroup 的空核远小于一次命令缓冲提交，GPU 时间戳量的是 encoder 间隙。这条臂只能证明"空 stage 不值得单独测"，不能当 floor |
| `kda small` 单 stage vs 三 stage，工作量相同 | 40.61 → 48.91 µs GPU | 每个多余 stage ≈ 4.2 µs（含 14 个 buffer 参数与 barrier） |
| `kda small` 三 piece vs 单 piece，同三 stage | 48.91 → 62.51 µs GPU | 切链本身 +13.6 µs |
| `kda paper` 单 stage vs 三 stage | 186787.82 → 186969.63 µs GPU | 每个多余 stage ≈ 91 µs（占 0.049%） |
| `kda paper` 三 piece vs 单 piece | 186969.63 → 187869.45 µs GPU | 切链 +900 µs，占 0.48% |
| 状态写+读 D=64，8 head（128 KiB ×2） | +64.6 µs（8 线程） | 4.1 GB/s：一个线程搬一个 16 KiB 状态，是延迟受限 |
| 同样的字节，2048 线程 | **+0.42 µs** | 630 GB/s |
| 状态写+读 D=8，2 head（512 B ×2） | +1.6 µs（2 线程） | 同理 |

每个数字是 5 次重复的中位数，跨度也在归档里：`small` 单 stage 39.7–41.0、单 piece 42.9–50.5、三 piece 60.6–64.1 µs；`paper` 三臂各自跨度 < 0.15%；D=64 窄拷贝 54.1–149.6 µs（宽拷贝 1.99–2.02）。小形状上 encoder 与计时粒度占比大，绝对值只能作分解参考，方向（三 piece 最慢、宽拷贝近零）在每次重复里都成立。

最后一组说明的是**拷贝本身的宽度效应**：同样的 128 KiB 状态，8 线程搬要几十微秒，2048 线程搬低于噪声。现有 fixture 每个 head 只有一个线程，跨 stage 交接时由这一个线程串行搬完整块 `D×D`，所以这一项在它身上确实昂贵。

**但它不能解释全部切链成本.** `paper` 的切链成本是 926 µs，而 512 KiB 的纯往返即使按探针量到的 4 GB/s 算也只有约 129 µs，差 7 倍；`small` 的 26.05 µs 对约 1–3 µs 的纯拷贝模型同样差一个量级。剩下的余量来自每个 stage 的 prologue（重读 `A_log`/`dt_bias`、重新载入并初始化 `state[D*D]`）、`paper` 上 `D*D = 4096` 的线程本地状态被溢出到设备内存后每 token 的访问代价，以及各 stage 的固定开销。本阶段**没有**把这部分分解干净，所以只能说"拷贝宽度是可测的一个杠杆"，不能说"切链成本就是拷贝成本"。

因此"SEQ 贵"不等于"协议贵"，但也不能简单归因成"把拷贝放宽就好"。真要压低这段成本，得先做上面那份 prologue/线程本地状态的分解，再决定改递推 kernel 的哪一段；改缓冲归属不会改变任何一项。

### 3.3 Host Program 的角色与成本

`kda` 的 planner 是候选自己的 native64 C99 Host Program，`input_dependent=true / preprocesses_inputs=true / inside_timing=true`，所以它进入候选计时边界。harness 记录的稳态耗时：

| 档 | Host wall（计时窗内，三臂区间） | 三臂之间的极差 |
| --- | ---: | ---: |
| `kda single` | 6.96 – 7.08 µs | 0.13 µs |
| `kda small` | 7.42 – 7.63 µs | 0.21 µs |
| `kda paper` | 21.35 – 22.08 µs | 0.73 µs |

也就是说：**切不切链几乎不影响 Host 成本**（planner 的循环最多 3 次，写 6 个 u32）。Host 的成本主要是映射输入的读取与 ABI 校验，`paper` 上更高是因为 4 个 `[N,H,D]` fp16 输入各 2 MiB。第一次调用另有约 320 µs 的一次性成本（见 host_runs 的 correctness 第一条），那是校验与建表，不是稳态调度。

### 3.4 状态容量随形状增长

容量来自 OpSpec，不是候选自选：

| 实例 | chain 数 | 每条状态 | 容量（状态数 / 字节） | 全切时实际写入 | 未用状态 |
| --- | ---: | ---: | --- | ---: | ---: |
| `kda:single` | 2 | 256 B | 4 / 512 B | 0 | 4 |
| `kda:small` | 2 | 256 B | 4 / 1024 B | 4 | 0 |
| `kda:tail` | 2 | 256 B | 4 / 1024 B | 4 | 0 |
| `kda:paper` | 8 | 16 KiB | 16 / 262144 B | 16 | 0 |
| `varlen_kda:fixed` | 2 | 256 B | 4 / 1024 B | 4 | 0 |
| `varlen_kda:mixed` | 6 | 256 B | 12 / 3072 B | 10 | 2 |
| `varlen_kda:uniform` | 4 | 256 B | 8 / 2048 B | 8 | 0 |
| `varlen_kda:capacity` | 8 | 256 B | 16 / 4096 B | 16 | 0 |

两个"未用"的行不是排版错误：`kda:single` 的 chain 只有 1 个 token，`min(3,1)=1`，**一个 handoff slot 都不会被写**；`varlen_kda:mixed` 里长度为 2 的段只要 2 个 piece，它的第二个 slot 留空。

## 4. 安全：不只是正常输入能过

`tests/backends/metal/test_seq_state_handoff_gpu.py` 在真实 Metal 上跑 27 个节点。正例覆盖登记的全部 case（`kda` 的三个：`normal`/`wide_range`/`outlier`；`varlen_kda` 的九个 L1 case 含 `zero_length`、`max_handoff_stress`、`chunk_boundary`），每个 case 都经过 harness 的三轮初值（out1/out3 毒化、out2 清零）。

负例按 harness 的检查顺序：

| 负例 | 构造 | 判定 |
| --- | --- | --- |
| 中/末 piece 从 `initial_state` 续传 | 把 handoff 读改成 `initial_state` | `incorrect` |
| 中/末 piece 从零续传 | 改成 `0.0f` | `incorrect` |
| 状态**部分写**（漏最后一行） | store 循环 `vi + 1u < D` | `unsafe` / `incorrect`（读到的行随 trial 初值变） |
| **旧 handoff 状态** | piece 0 在 slot 1 种入 `initial_state + 0.5`，piece 1 整段跳过 store | 拒绝；误差 4232.65 随种入偏移（0.25 → 2995.36）改变，证明末 piece 真读了这份字节；只种不跳的控制实验为 `ok` |
| consumer-first（交换 stage 0/1） | 候选几何 stage 顺序 | `incorrect` / `unsafe` |
| 少一个 stage 但仍是三 piece | 几何只留 2 stage | `unsafe`：`final_state` 没人写 |
| 发布的 piece 数多于 dispatch 的 stage | `item_counts = 4` | `unsafe`：fixture 自己的 `published > 3` 守卫让所有 stage 直接返回，`final_state` 没人写 |
| **work list 越界** | planner 发布 4 个 piece，写满 8 个 u32 > 容量 6 | 拒绝，原因含 `guard sentinel` |
| `final_state` 部分写 | 末行不写 | `unsafe` |

`varlen_kda` 侧另有既有节点固定 flag 协议（未发布 READY、未清 flag、跨链共用 slot、依赖零初始化 flag、旧 READY + 旧 state 的三控制实验、空序列不推进状态）与设备 work list 与 NPU adapter 的逐字节差分，本阶段原样复跑。

### 4.1 `out1/out2/out3` 与 fresh

每个正例都跑三轮初值，`safety.ok=true` 且 `nondeterminism` 为空，说明结果不依赖输出初值与 `read_write` 状态，只读输入未被改。八个被测 arm 全部拿到 fresh confirmation：identity `match`、`status=confirmed`，fresh 种子由 P2-I 的公开 master key 派生，没有一个是搜索种子 7。fresh subject 在 `evidence/confirmations.json`。

注意 `tasks/kda` **没有 `ieee` case**（它的 `reference.py` 只生成 `normal`/`wide_range`/`outlier`）。本阶段没有加：给任务加 case 会移动任务合同身份，属于另一个阶段。所以"全部声明 case"在这份任务是那三个，没有放宽任何门。

## 5. 要不要给 SEQ 新增 ScratchSpec

答案：**不要。** 但理由必须说准，因为初版这里写错过一条，记录保留在下面。

**初版错在哪（已撤回）。** 初版写"因果状态天然是 `read_write`，所以 `uninitialized_poison` 装不下"。独立复核给出反例并被本阶段复现：把状态改成**两块缓冲乒乓**（stage 0 写 `state_a`，stage 1 读 `state_a` 写 `state_b`，stage 2 读 `state_b`），两块都声明 `uninitialized_poison`，`preflight_submission` 返回 **True**。也就是说"每块缓冲的首处使用必须是写"这条规则是可以满足的，协议**能够**表达一次因果状态交接。初版那句结论是错的，已从代码断言、本文与 `AGENTS.md` 中删除。

现在的四条理由：

1. **没有可换的性能。** 同一批字节放在 OpSpec workspace 还是候选 scratch，dispatch 次数、barrier 次数、状态大小都不变。§3.1 的 A/B 已经把切链成本量出来了，它来自"切了几段"和"谁搬状态"，与缓冲归属无关。
2. **能同时满足严格 poison 规则与因果交接的形态要重构状态布局，而且不省任何东西。** 单块缓冲同时承担"接收"和"交出"，首处使用必然是 `read_write`，被生命周期直接拒绝（真实消息见下）。乒乓形态合法，但它**不改变总字节**：两块 `H*D*D` fp32 加起来正好等于声明的 `2*H*D*D` 容量（`paper` 上各 128 KiB，合计 256 KiB = 工作区 262144 B），所以既没有存储惩罚也没有流量收益，只是多一个 kernel 参数、换一套 slot 布局。

   ```text
   execution_geometry.scratch: uninitialized_poison scratch 'chain_state' is first used as
   read_write; the first active stage must bind it write-only so a later stage cannot read an
   unwritten value
   ```
3. **`device_zero` 只多付一次计时内清零，且换不到探针。** 它不比 workspace 严格：正确性对 `device_zero` scratch 同样在 out1/out3 毒化、out2 清零，而残留/全轮探针只给 `uninitialized_poison`。这条路线既没有性能收益，也没有验证收益。（本阶段没有测到它的计时成本：它和乒乓形态一样，先被下面的 worker 缺陷挡住。）
4. **OpSpec workspace 把容量写进了 ABI，这比候选自选更强。** harness 因此可以独立算出 `handoff_states` 有多大、检查越界、并在参考侧对 `final_state`/`read_write` 建立对照。搬到 scratch 会让"一条链最多 3 段、每段状态多大"从合同事实退化成候选声明。

### 5.1 现存限制：bundle + Host Program + scratch 目前跑不起来

乒乓形态与 `device_zero` 形态都能过预检（单块 `read_write` 形态连预检都过不了），但前两者真正求值时都停在同一个地方：

```text
v2 worker failed: TaskContractError: the lifecycle was registered for a different resolved geometry
```

根因是 `tcenv/v2_worker.py::_lifecycle_from_document` 重建 `ResolvedGeometry` 时没有带 `scratch` 字段：`ResolvedGeometry.document()` 在 scratch 非空时会写出该键（`tcenv/track/geometry.py`），于是控制机冻结的几何文档与 worker 重建的不相等，`tcenv/taskcontract/runner.py` 的身份比较就失败。`msl-single`（conv2d 走的路径，没有 Host Program、没有 lifecycle 对照）不受影响，所以 P2-I 的 conv2d scratch 能跑。

这条不是本阶段的决策依据，但它意味着**任何** `msl-bundle + Host Program + scratch` 候选现在都无法求值，包括乒乓形态的 SEQ。修它要改 `tcenv/v2_worker.py`，会移动 79 文件闭包并使 P2-N/O/P 已登记的摘要失效，因此按"不混阶段"的约定只记录，留给后续阶段。两种形态的真实结局都在 `evidence/seq-handoff-scratch-attempt.json`。

## 6. 采用路线

**保留现有实现：** `handoff_states` / `work_list` / `item_counts` 继续是 OpSpec workspace，多 stage 仍用串行 encoder + buffer barrier 定序，候选仍拥有切分与 stage 顺序，Host Program 仍是唯一能读本次输入写 work list 的角色。本阶段没有改任何 ABI、没有改提示面、没有重签 trust、没有动六 RPC/字段号/HMAC/BlobChannel。

什么时候值得重新考虑 SEQ：

- 任务形状让 grid 远小于核数，**且** 同一条 chain 的多个 piece 能真的并发（需要新的同步原语，不是 scratch 能解决的）；
- 或者先把 §3.2 未分解的余量（每 stage prologue、溢出的线程本地状态）拆开，再判断要不要让递推 kernel 多线程搬状态；
  单看拷贝那一项，放宽宽度能把 handoff 降到噪声，但按归档它只解释得了切链成本的一小部分；
- 两者都不是"换个缓冲声明"能做到的。

## 7. 复现

```bash
# 三臂测量 + 八次 fresh confirmation + scratch 尝试 + probe + 决策（约 15 分钟 GPU）
uv run python -m tests.backends.metal.seq_handoff_record

# 只重算某一项
uv run python -m tests.backends.metal.seq_handoff_record --measurements
uv run python -m tests.backends.metal.seq_handoff_record --decision
uv run python -m experiments.seq_handoff_probe

# 验收
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/verify.py
uv run pytest -q -n 0 tests/shared/test_seq_handoff_decision.py
uv run pytest -q -n 0 -m gpu tests/backends/metal/test_seq_state_handoff_gpu.py
```

## 8. 剩余限制

- 全部数字只覆盖登记形状与 fp16、本机 Apple M4 Max。`kda:paper` 的单条链仍由一个线程递推（186 ms 量级），所以 `paper` 的"0.5%"只说明 handoff 相对一次慢递推很小，不能外推成"SEQ 在 paper 上便宜"。
- `tasks/kda` 没有 `ieee` case；非有限输入的行为不在本阶段的合同证据里。`mamba2_chunk_scan` 有 `ieee`（对 `states` 注入），但它不使用 SEQ，所以不能替代。
- 空 stage 的 barrier 成本在本机**不可分辨**（斜率 ±0.2 µs、甚至为负）：8 个 threadgroup 的空核比一次命令缓冲提交还小，GPU 时间戳量的是 encoder 间隙。§3.2 因此改用"同工作量、单 stage 对三 stage"来量 stage floor，空 stage 那条臂只作为"此路不通"的记录。
- 三臂的 full run 之间仍有跨 run 噪声；本阶段用 3 次独立 full run 取中位数、并要求 cv < 6%，但没有做同轮配对。同轮配对的证据是 probe 的同进程臂（§3.2），两者方向一致、量级不同（harness 端到端含每 stage 的参数编码）。
- `varlen_kda` 的 `capacity` / L2 档没有纳入 SEQ 消融，只有 L1 的 `mixed`。
- 没有跑真实 NPU；清微仍只有模拟器（`simulated=true`、`publication_allowed=false`）。

## 9. 总结：什么时候重算优于存储与通信

存储与通信不一定是唯一瓶颈，但它们经常决定复用是否划算。选择算法时比较的是两条路径的总成本，而不是“复用”或“重算”哪个名字更高级：

```text
重算成本
    vs
中间结果写入 + 读取 + 同步 + 额外 stage + 存储占用 + 调度成本
```

当重算更便宜时，就重新计算；只有复用节省的重复工作超过全部存储与通信成本时，才外化中间结果。能留在寄存器或 threadgroup memory 的，优先片上复用；全局 scratch 必须有足够大的真实复用收益。

本项目的正反例：

| 场景 | 结论 | 原因 |
| --- | --- | --- |
| `gemm_silu` | REG 重算/融合更优 | 中间结果只用一次，全局写读没有减少任何计算 |
| `bitdelta_gemm` | tile 内现场解包更优 | 展开权重减少重复解码，但展开流量、producer stage 和 barrier 更贵 |
| `conv2d small` | 单阶段重算 U 更优 | 两阶段启动与同步已超过 MLX 总时间 |
| `GEMM Split-K` | 不切分更优 | 输出并行度足够或已接近 roofline，partial 写读和归并没有收益 |
| `KDA / varlen SEQ` | 不为统一而新增 ScratchSpec | 串行 stage 切链没有新增并发，handoff 与调度是纯支出 |
| `conv2d paper` | **全局 U 复用成功** | 同一份 U 被 6272 个输出 tile 共享，节省的重复变换超过全局流量，fresh 617.29 µs 对 recompute 1474.42 µs，快 2.3885× |

因此最终原则不是“存储和通信一定是瓶颈”，也不是“缓存一定更快”，而是：**先找真实复用，再比较完整成本；没有净收益就重算，有明确复用收益才存储。**
