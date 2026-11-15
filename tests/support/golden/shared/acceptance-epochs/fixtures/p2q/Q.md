# P2-Q：KDA / packed-varlen 的 SEQ 顺序状态传递验证

- epoch：`p2-q-seq-state-handoff-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`；NPU 仍 simulation-only

本阶段只做 Plan 2C。P2-I/J/K/L/M 已提交；P2-N/O/P 的未提交工作树原字节保留，本阶段没有修改它们引用的任何文件，也没有 `git commit/reset/clean`。

## 结论

**现有 SEQ 协议是安全的，但在这套交付里切链是纯支出：`handoff_states` 继续留在 OpSpec workspace，不新增 ScratchSpec，不改 ABI，不改实现。**

三条独立结论：

1. **安全。** 状态来源（initial / zero / 旧 slot）、状态存储（部分写）、stage 顺序（consumer-first、少一个 stage、发布数多于 stage 数）、work list 越界、`final_state` 部分写，全部在真实 Metal 上被拒绝，且分类清楚（`incorrect` 或 `unsafe`）。正例覆盖登记的全部 case，每个 case 跑三轮初值，`safety.ok`、无非确定性；八个 arm 全部通过 fresh confirmation（identity `match`，fresh 种子均不等于搜索种子 7）。
2. **成本。** 三臂同算法对照（device 源码逐字节相同，只改 Host 的 piece 数与候选几何的 stage 数）：`kda small` 三 piece 247.44 µs vs 单 stage 201.13 µs（**+23.0%**），`tail` +19.1%，`paper` +0.5%，`varlen mixed` **+56.4%**。分解为 handoff 与 stage floor 两项，前者在 `small` 是 26.05 µs、在 `paper` 是 926.16 µs。
3. **拷贝宽度是一个可测的杠杆，但它解释不了全部切链成本。** 同样的 128 KiB 状态往返，8 线程中位约 66 µs（5 次样本 54–150 µs，4–5 GB/s），2048 线程相对对照只多 0.42 µs（>600 GB/s，本次归档）。但 `paper` 的切链成本 926 µs 与 512 KiB / 4 GB/s ≈ 129 µs 差 7 倍，`small` 的 26.05 µs 与纯拷贝模型差一个量级；余量来自每个 stage 的 prologue、`D*D` 线程本地状态被溢出后的访问代价和 stage 固定开销。本阶段没有把这部分分解干净，因此不改递推 kernel。

## 为什么不新增 ScratchSpec

| 理由 | 证据 |
| --- | --- |
| 性能上没有可换的东西：同样字节、同样 dispatch、同样 barrier | 三臂 A/B；probe 的宽窄拷贝对照 |
| 单块缓冲同时"接收并交出"状态时首处使用必然是 `read_write`，被生命周期拒绝 | 真实 `GeometryError`：`first used as read_write … must bind it write-only` |
| **合法的乒乓形态不省任何东西**：两块 `H*D*D` fp32 合计正好等于声明的 `2*H*D*D` 容量（small 1024 B = 1024 B，`storage_penalty=false`），却多一个 kernel 参数并换一套 slot 布局 | `evidence/seq-handoff-scratch-attempt.json` 的 `two_buffer_pingpong`：`preflight_submission = True`、`scratch_bytes == workspace_bytes_at_the_same_capacity` |
| `device_zero` 不比 workspace 严格：`device_zero` scratch 在正确性上同样 out1/out3 毒化、out2 清零，残留/全轮探针只给 poison；它只多一次计时内清零 | 三种初值表（`docs/架构/中间结果复用与Scratch Workspace.md`）；本阶段没有测到它的计时成本，因为先被下面的 worker 缺陷挡住 |
| OpSpec workspace 把容量写进 ABI，比候选自选更强：harness 能独立算容量、查越界、对 `final_state`/`read_write` 建对照 | `state_algebra` 的八个实例 |

**初版这里写错过一条，已撤回并保留记录。** 初版断言"因果状态天然 `read_write`，所以 poison 装不下"，独立复核给出反例（上面的乒乓形态）并被复现，因此现在的理由只到"单块缓冲不合法、合法形态不省任何东西却要改布局、且目前跑不起来"。

**现存限制（不是本阶段引入，也不是决策依据）：** 乒乓形态与 `device_zero` 形态都能过预检，但 `msl-bundle + Host Program + scratch` 真正求值时停在 `TaskContractError: the lifecycle was registered for a different resolved geometry`。根因是 `tcenv/v2_worker.py::_lifecycle_from_document` 重建 `ResolvedGeometry` 时没有带 `scratch` 字段，控制机与 worker 的几何文档因此不相等。`msl-single`（conv2d，无 Host Program）不受影响。修它会移动任务包闭包，使 P2-N/O/P 登记的 79 文件摘要失效，所以不在本阶段做。完整记录在 `evidence/seq-handoff-scratch-attempt.json`。

## 三臂实测（搜索种子 7，各 3 次独立 full run，取中位数）

| 档 | A 三 piece / 三 stage | B 单 piece / 三 stage | C 单 piece / 单 stage | A 相对 C | handoff (A−B) | stage floor (B−C) | cv(A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kda single` | 153.81 µs | 152.54 µs | **137.65 µs** | +11.7% | +1.27 µs | +14.89 µs | 0.023 |
| `kda small` | 247.44 µs | 221.40 µs | **201.13 µs** | +23.0% | +26.05 µs | +20.26 µs | 0.023 |
| `kda tail` | 253.01 µs | 227.80 µs | **212.37 µs** | +19.1% | +25.20 µs | +15.44 µs | 0.020 |
| `kda paper` | 188402.36 µs | 187476.21 µs | **187375.16 µs** | +0.5% | +926.16 µs | +101.05 µs | 0.0011 |
| `varlen mixed` | 53.72 µs | 34.35 µs（4 stage 不变） | — | +56.4% | +19.36 µs | 无 | 0.044 |

`kda single` 的 N=1 只有 1 个 piece，A 与 B 是同一份 work list：两条臂相差 0.8%，说明差异确实来自切链。

## 证据与验收

`evidence/`：

- `seq-handoff-measurements.json`：三个 `kda` 档 + `varlen mixed` 的多臂测量，含每次 full run 的 cv、case、dispatch、workspace 字节、Host 稳态耗时与源摘要；
- `confirmations.json`：八份 fresh subject（完整 search + confirmation 报告与 identity）；
- `seq-handoff-probe.json`：空 stage 斜率、相同字节窄/宽拷贝对照、`kda` 单 stage/单 piece/三 piece 的同缓冲对照，`not_a_candidate=true`；
- `seq-handoff-scratch-attempt.json`：三种尝试的真实结局（单块 `read_write` 被生命周期拒绝；两块乒乓合法但被 worker 几何身份挡住；`device_zero` 同样被挡住且计时成本未测）；
- `decision.json`：由上述文件重算的路线决策，verifier 逐字段重算；
- `review.json`：独立复核三轮的记录（r1、r2 都被拒绝，两条 blocking 原文保留；r3 给出 `done=true`）。

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/verify.py            # 18 CPU + 27 真实 Metal
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/verify.py --static   # 只跑 CPU 节点
```

预期最终行：`P2-Q acceptance (18 CPU + 27 real Metal nodes; SEQ protocol safe, handoff not worth a ScratchSpec, publication_allowed=false)`。

重录：

```bash
uv run python -m tests.backends.metal.seq_handoff_record                      # 全部（约 15 分钟 GPU）
uv run python -m tests.backends.metal.seq_handoff_record --measurements
uv run python -m tests.backends.metal.seq_handoff_record --confirmations
uv run python -m tests.backends.metal.seq_handoff_record --scratch
uv run python -m experiments.seq_handoff_probe
uv run python -m tests.backends.metal.seq_handoff_record --decision
```

## 全量安全 CPU 套件的失败分类

`uv run pytest -q -m 'not gpu and not llm and not slow'` 本轮是 **5175 passed / 60 failed / 14 skipped**。失败只落在三组已知类别，且**没有任何一个失败文件引用本阶段新增的模块**（`seq_handoff`、`seq_handoff_record`、两个测试文件）：

| 类别 | 节点数 | 真实原因 |
| --- | ---: | --- |
| 已知 P2 epoch 期望身份 | 26 | `test_shape_portfolio_manifest/resume/confirmation` 与 `test_triton_baseline`：`expected identity is not independently rebuilt from this pack/probe`。与 `AGENTS.md` 记录的“26 个已知 P2 epoch”一致 |
| selfcheck AF_UNIX | 22 | `OSError: AF_UNIX path too long`（socket 路径长度限制） |
| 进程权限 `ps` | 8 | `PermissionError: [Errno 1] Operation not permitted: 'ps'`，`ps aux` 在本机直接返回 126 |
| 其余环境/工具链 | 4 | `test_hw_features`、`test_kernel_agent`、`test_track_mt1b_build_seam` 的既有环境失败 |

本阶段新增的 18 个 CPU 节点与 27 个真实 Metal 节点全部通过。这三类必须单独分类，不能写成“全绿”。

## 边界

- `publication_allowed=false`：所有倍数与微秒数是本机本协议的相对位置，不是生产 speedup。`kda` 的 `paper` 档候选比登记的 `torch_mps` 逐 token 基线慢约 8 倍，本阶段不安慰性引用它。
- 没有启动 `taic` 付费搜索、没有模型调用；没有改提示面/`current_prompt_sha1`、六 RPC、字段号、HMAC、BlobChannel；没有重签 trust。
- 清微 NPU 仍只有模拟器，本阶段没有跑真实 NPU，也没有改 `tests/support/npu_sim/`。
- P2-I..P 的 manifest 逐文件不变。任务包闭包仍是 79 文件，摘要 `5512395852…`。
- `tasks/kda` 没有 `ieee` case；本阶段没有加（会移动任务合同身份）。

## 剩余限制

- 三臂是三次独立 full run 的中位数，不是同轮配对；同轮配对的旁证是 probe 的同进程臂，两者方向一致、量级不同（harness 端到端还含每 stage 的参数编码）。
- 空 stage 的 barrier 成本在本机不可分辨（斜率 ±0.2 µs、甚至为负）：空核比一次命令缓冲提交还小，量到的是 encoder 间隙，因此不当作 floor。
- `kda paper` 的单链仍由一个线程递推（186 ms 量级），"0.5%"只说明 handoff 相对一次慢递推很小，不能外推成"SEQ 在 paper 上便宜"。
- `varlen_kda` 只做了 L1 `mixed` 的 SEQ 消融，L2 `capacity` 档没有纳入。
- 全部数字只覆盖登记形状与 fp16、本机 Apple M4 Max。
