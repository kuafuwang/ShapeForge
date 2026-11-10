# V6 chunked/matrix 真实规模验收

> **归档位置与当前可核验性（2026-11-10）**：这份包原来在 `docs/架构/v6-chunked验收/`，
> 按“验收证据进 golden”的既有约定搬到此处；字节未改。现在 `verify.py --check` **不通过**，
> 报 `confirmation seed is not derived from this validation plan`：归档的 freeze 是
> `67cd0926…`、validation plan 是 `765fb710…`，而当前代码从同一道题重建出来的是
> `660d06d4…` / `e7278968…`，两档 seed 因此全部对不上。这与 `P2/F.md` 记录的
> `portfolio_freeze_digest` 同一类漂移，不是本目录搬动造成的；修它要给六档重录 sidecar，
> 需要真实 Mac 与显式授权。在此之前这里只是**历史测量记录**，不是当前可重算的证据。
> 现行的正确性固定点是 `tests/backends/metal/test_varlen_v6_chunked_gpu.py`。

## 结论与边界

V6 的六档真实 Metal 验收已完成：D=128，H=64/96，fixed/mixed/uniform，
`total_tokens=8192`。每档都有：

- canonical 及全部 9 个 L1 case 的 output + 每序列 `final_state` 正确性；
- search 与不同 fresh seed 的独立执行；
- 15 次 fresh E2E 计时，candidate/baseline cv 均不超过 0.05；
- build/evaluation identity 完全匹配；
- benchmark + golden 两个用途点的 confirmed sidecar；
- 重新激活成功的独立 DispatchManifest。

候选是 `tests/support/golden/metal/varlen/chunked_gform.metal`，**交付格式为
`msl-single`，不是 msl-bundle**，不含 Host Program，也没有调用付费模型搜索。
candidate SHA-256：

```text
4e724ba69be0d28262fe758b4c120e3a6ca1d8d0e49c150724258a46580517e4
```

本目录是新的 V6 evidence，不改写 R0–R6 或既往验收包。后续安静窗口的 P2 最终重录见
`shape-portfolio-阶段验收/P2/F.md`；本结论不是 P3-B/P3-C 或 NPU 实机结论。

## 算法与正确性修复

候选每个 `(sequence, head, v-tile)` 使用一个 threadgroup，BT=8、D=128，使用真实的
`simdgroup_float8x8` 计算 G-form 的 `A` 和 `Base`。下三角 residual 求解之后，仍按 oracle 的
fp32 token 顺序更新 state/output；因此它是真实的 chunked/matrix 候选，但没有声称整个
recurrence 都已矩阵化。

独立审查定位的关键错误是：oracle **先对当前 token 的 state 施加 decay，再计算 residual**。
旧实现把 `KD/QD` 写成 `K/Q * exp(L_prev)`，漏掉当前 token decay。现实现使用：

```text
Kbar = K * exp(-L)
KD   = K * exp(L)
QD   = Q * exp(L)
```

原始 fp32 state recurrence 仍使用 raw K/Q。`tests/backends/metal/test_varlen_v6_chunked_gpu.py`
同时固定：小规格全部 9 个 L1 case 通过；把 KD/QD 改回 `L_prev` 的候选被判 `incorrect`；
六档真实规模 canonical 全部通过。没有修改 oracle、容差或 case 域。

## Fresh E2E 性能

下表直接读取各档 `results/<size>.json` 中的 fresh confirmation。candidate 单位为 ms，
baseline 单位为 s；speedup 是 `baseline.median_s / candidate.median_s`，不是最小值之比。

| 档位 | candidate ms | candidate cv | torch_mps s | baseline cv | speedup | active manifest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v6_h64_fixed | 176.145 | 0.0328 | 1.745453 | 0.0032 | 9.9092× | `394a2304356e…` |
| v6_h64_mixed | 169.027 | 0.0391 | 1.754849 | 0.0027 | 10.3821× | `5ab95232f7c8…` |
| v6_h64_uniform | 178.351 | 0.0473 | 1.755228 | 0.0045 | 9.8414× | `8aad0f8c83ca…` |
| v6_h96_fixed | 264.772 | 0.0432 | 2.308941 | 0.0032 | 8.7205× | `a5c5feefc1fb…` |
| v6_h96_mixed | 252.930 | 0.0262 | 2.313761 | 0.0040 | 9.1478× | `fa7bc6f5e772…` |
| v6_h96_uniform | 248.438 | 0.0414 | 2.317038 | 0.0050 | 9.3264× | `d82bbd189566…` |

每一档独立超过 1.10 gate，没有用 geomean 隐藏局部回归。完整 digest、15 个原始样本、
fresh seed 和身份字段见 `results/`、`confirmations/` 与 `manifests/`。

### 为什么比值接近 10×，以及它不代表什么

当前唯一登记基线是 `tasks/varlen_kda/baselines.py` 的 `torch_mps`：它在 Python 中逐 sequence、
逐 token 调用 Torch-MPS 的 norm/sigmoid/exp/reduction/state update。8192 个 token 会产生大量
框架调用与 MPS dispatch。候选把整条 `(sequence, head, v-tile)` 链融合进一个 Metal kernel，
并使用 simdgroup matrix。`BaselineRunner.run_k` 对 Torch 在函数调用前开始墙钟计时、结束后
`torch.mps.synchronize()`，candidate 同样走 v2 E2E runtime，因此比值包含真实框架/launch 开销。

所以这些数字**只证明相对该朴素 Torch-MPS 逐 token 基线的加速**，不证明相对专门优化的
KDA、CAKE2、CUDA/CANN 或其他 vendor kernel 也有同样加速。没有 NPU 模拟数字参与比较，
也没有把 device-only 时间冒充 E2E。该 msl-single 路径没有 Host Program；报告没有另造一套
host/device 分解来替代 E2E 谓词。

六次 benchmark 均为 `provider=candidate`、`fallback_id=null`、`credits_variant=true`，
`identity={kind:match,differences:[]}`。search seed=7，fresh seed 各不相同；每次 fresh
correctness 都覆盖 9 个 L1 case。fallback 的正确性和时间没有被混入这些 candidate 数字。

## 为什么完整确认约一小时

每档 benchmark 用途做 search + fresh 两次完整评估，golden 用途再做 search + fresh 两次
正确性评估。四次评估都运行 9 个大规模 case；bench 还包括 candidate/baseline 交错计时及必要的
安静重试。下表把四次报告的 timeline 相加：

| 档位 | 四次 correctness 合计 s | search+fresh bench 合计 s | 四次评估总计 s |
| --- | ---: | ---: | ---: |
| v6_h64_fixed | 363.9 | 151.1 | 515.0 |
| v6_h64_mixed | 369.4 | 87.8 | 457.2 |
| v6_h64_uniform | 371.0 | 151.9 | 522.9 |
| v6_h96_fixed | 499.6 | 116.7 | 616.3 |
| v6_h96_mixed | 504.2 | 232.4 | 736.6 |
| v6_h96_uniform | 506.2 | 134.9 | 641.1 |
| **合计** | **2614.3** | **874.8** | **3489.1（58.2 分钟）** |

因此一次六档命令被 3600 秒外层 timeout 中止是时间预算刚好不足，不是 hang 或数值失败。
单档续跑不会覆盖其他档：`results/` 逐档保存，最后由全部六份结果合成 `summary.json`。

## 文件与重放

- `verify.py`：真实 GPU 确认、逐档 sidecar/result/manifest 写入及汇总；不调用模型、不改 P2。
- `confirmations/<size>.{benchmark,golden}.json`：完整 search/fresh subject 文档。
- `manifests/<size>.json`：active DispatchManifest，含两个用途点。
- `results/<size>.json`：从 confirmed sidecar 提取的逐档计时与身份摘要。
- `summary.json`：只有六档都 active 才为 `status=confirmed`。

只读重放（不执行 GPU）：

```bash
uv run python tests/support/golden/shared/varlen-v6-chunked验收/verify.py --check
```

该命令从当前 task pack、候选字节、master-seed 派生计划和归档 sidecar 独立重建 expectation，
重新激活并逐字节比较 manifest。六档均已通过，digest 与归档完全一致。

真实 GPU 重录（需要单独授权，约一小时）：

```bash
uv run python tests/support/golden/shared/varlen-v6-chunked验收/verify.py
# 或只续跑一档：
uv run python tests/support/golden/shared/varlen-v6-chunked验收/verify.py --sizes v6_h96_uniform
```

## 回归与未扩展的结论

- V1 + V2 Dyn-0 + native Host + V5 + V6 真实 Metal：51/51 passed。
- 定向 CPU/协议/结构回归：633 passed。
- P2 安静窗口重录后，完整 CPU 快速套件：**4549 passed、13 skipped**。没有删除、skip/xfail
  或放宽既有测试。
- Dyn-0 保持默认。V6 候选 grid 精确对应 `(sequence, head, v-tile)`，没有引入 Dyn-1。
- 固定长度 KDA 标准任务与复杂 bundle 的离线/CPU 示例见 `docs/架构/fixed-kda-bundle示例.md`；
  真实 Metal 与付费 msl-bundle 搜索未启动，Qingwei NPU-P0 仍 blocked。
