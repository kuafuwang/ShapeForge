# 在 Apple GPU 上做 AI Lowering：Triton → Metal Shading Language

本文是 [README.md](README.md) 的中文译本。命令、路径和配置名与英文版一致。

这是论文 *"AI as a compiler: compiling Triton kernels without the Triton compiler"* 的实验设置，在 Apple M4 Max 上的再实现。
大模型把 Triton 内核直接翻译成 Metal（MSL）内核；评测环境对每个候选做编译、安全检查、差分测试、基准测试和性能剖析，搜索过程只保留安全、正确且更快的候选。

| 论文 | 本仓库 |
| --- | --- |
| Triton → PTX | Triton（作为规格文本）→ MSL |
| TCEnv（环境） | `tcenv/` |
| TAIC（智能体） | `taic/` |
| ptxas + CUDA driver | `MTLDevice.newLibraryWithSource`（PyObjC） |
| compute-sanitizer | 保护带 + 用 NaN 投毒的输出 + 确定性 + `MTL_SHADER_VALIDATION` |
| Nsight Compute | 流水线统计、GPU 时间戳、实测 roofline、可选的 `xctrace` |
| 自动调优的 Triton 基线 | MLX 与 PyTorch-MPS 中较快的一方；可选地由 `triton-msl` 编译的 Triton（见下文） |
| FP8 | Apple GPU 上不可用；改用 FP16 和 BF16 |
| GPT-6 Astra、美元预算 | 任意 OpenAI 兼容端点（默认 `deepseek-flash`），带缓存感知的美元账本 |

## 快速开始

```bash
uv sync
uv run python -m tcenv.cli tasks                       # 12 个内核（+ 第二阶段：flash_attention、lion、bitdelta_gemm）
uv run python -m tcenv.cli naive gemm                  # 测试框架自检：一个手写内核走完整条流水线
uv run python -m tcenv.cli eval softmax my_kernel.metal
uv run python -m taic.cli run softmax --budget 1.0     # 让大模型 lowering 一个内核，这里只给 $1 冒烟预算（默认是论文的 $15，见「预算」）
uv run pytest -q                                       # 环境、编排器（脚本化的大模型）、实验脚本
```

配置在 `taic.toml` 里，论文的搜索预算由 `paper` 预设 `taic.paper.toml` 叠加在其上（见「预算」）；凭据从不写入仓库（从该文件所指的路径读取，或从 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 读取），并会在每一份记录里被打码。

## GPU 安全（请先读）

Apple GPU 同时驱动显示器，**正在运行的 Metal 内核无法被抢占**。一个不终止的内核曾把这台机器卡死到必须硬重启。因此：

* 不要手动运行带无界循环的内核，也不要添加会这样做的测试；
* 任何代码到达 GPU 之前，测试框架会拒绝 `while (true)` / `for (;;)` / 非 Metal 的 `#include`（`tcenv/screen.py`）；
* 每次评测都在工作进程里运行，该进程在每次 GPU 启动前后打心跳；若单次启动已在途超过 2 秒（开启着色器校验时为 4 秒），父进程会对该工作进程发 `SIGKILL`，并强制执行总超时；
* 第一次评测超时后，编排器会停止**整次运行**（`max_hangs = 1`）；
* 真正的挂起测试默认跳过，除非设置 `TCENV_ALLOW_GPU_HANG_TESTS=1`；监督逻辑改由假工作进程覆盖。

## 测量卫生

并发任务会扭曲计时：套件还在跑的时候测一次 triton-msl，结果会慢 2–4 倍；一个变异系数（cv）为 16% 的 GEMM 候选（安静运行时为 0.5–1.5%）曾挤掉更好的内核。因此：

* 本项目里的每一次计时（候选基准、基线、triton-msl、标定、xctrace 录制）都独占一把机器级锁；评测器的其他 GPU 工作则以共享方式持有同一把锁。这样不同实验可以同时跑，测量时段却不会重叠（`tcenv/gpulock.py`，对 `~/.cache/ai_lowerling/gpu.lock` 做 `flock`，可用 `TCENV_GPU_LOCK_FILE` 覆盖）。等锁的时间不计入评测超时。本项目之外的程序不受这把锁约束；
* 单次重复的变异系数超过 5% 的基准会重测（最多 3 次，保留最安静的一次，其余由 `benchmark.noisy` 标记）；
* 候选要替换当前最优，必须同时快过 `min_improvement`，并且快过两次加速比的 `noise_z` 个标准误；
* 结束时，最优者以及在噪声范围内与它并列的最多 `confirm_top - 1` 个候选会被重新测量；最快的成为结果（`result.json` 的 `confirmations`）。

## 预算

论文给每次搜索设了固定的美元上限（arXiv 2609.36800v1，`docs/2609.36800v1.pdf`）：第 4.3 节和表 1，「卷积、矩阵乘、FlashAttention 每次运行最多 $15，较简单的逐元素 SwiGLU 内核 $1」；第 4.4 节和图 5，十个已发表内核「在 $15 预算下」。搜索一直重复到预算耗尽（第 3.3 节）；预算在每次 LLM 请求之前检查，所以已经发出的请求可以略微超出上限。本仓库照此办理，**默认预算是每次运行 $15**（`SearchConfig`、`taic.cli run`、`run_suite`、`repeatability`、`p1_runs`）。图 4 的 12 内核网格没有写明上限，同样用 $15；重复性实验（表 1）像论文一样给 SwiGLU $1。

* **配置在哪。** `taic.paper.toml` 是叠加在 `taic.toml` 之上的预设（设了 `budget_usd = 15` 和 `max_rounds = 100`：$15 时结束运行的是预算而不是轮数，原来的 12 轮上限会在花掉约三分之一的钱时就停下）。`--preset base` 或 `TAIC_PRESET=base` 只读 `taic.toml`；`--config FILE` 原样使用。预设把 `patience`（连续 N 轮没有新最优就停）从 4 提到 12：短暂的平台期不会截断 $15 的运行，真正收敛的运行仍会提前结束，所以不再进步的内核花不到 $15。
* **覆盖。** 到处都有 `--budget USD`，`run_suite` 和 `repeatability` 还有 `--budget-scale F`（例如 `0.2` 做冒烟）。语义测试（每次 $0.15）、`counter_teaching`（$1）和单元测试特意保留小预算。
* **最坏情况。** 若每次运行都花满预算：`run_suite --tasks all --dtypes fp16,bf16` 是 24 次运行、$360；`repeatability` 是 5 次 x（$15 + $15 + $1）= $155（加上 `flash_attention` 为 $230）；`p1_runs` 是 11 次运行、$165。这三个驱动都有可选的 `--max-total-usd USD`（默认关闭）：整次调用放不下的运行会被跳过。目前没有任何运行接近上限：已结束运行里花得最多的是 $3.12（8 轮 FlashAttention，被 $3 预算截停）。
* **更早的结果不可比。** 它们用的是更小的预算，看板逐运行显示（每个 `result.json` 也记录 `budget_usd`）：`runs/suite_full` 每个内核 $0.5–$1.2，`runs/repeat_*` 每次试验 $3，`runs/p1` 多数 $3（后来启动的 `*_paper` 运行是 $15），`runs/semantic_main` $0.15，`runs/pilot_suite_mini_old_protocol` $1.25–$1.5。在那里以 `budget_exhausted` 结束的运行，在论文预算下本会继续搜索。

## 实验（论文第 4 节）

```bash
uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16     # 图 4：加速比相对花费，12 个内核（每次 $15；冒烟用 --budget-scale 0.2）
uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5   # 表 1（$15，swiglu $1）
uv run python -m experiments.semantic_tests --n-random 10                 # 第 4.1 节
uv run python -m tcenv.cli profile gemm best.metal                        # xctrace：spill、代码体积、热状态
```

怎么读这些数字：一次运行报告的加速比是**确认后**的加速比（胜者及其并列者用新种子、两倍重复次数重新测量）。套件表还会列出该任务手写朴素内核相对同一基线的加速比，因为有些基线偏弱（例如 Lion 和 BitDelta GEMM 没有融合的库实现）。朴素内核已经胜过基线时，收益里有一部分来自基线本身偏弱。

## 看板与预算告警

`uv run python -m observe` 启动只读看板（默认 `http://127.0.0.1:8765`；改了代码要重启，页面在启动时读入）。它把每次运行的预算和已花费并排显示，并对越过预算的运行告警。规则是一个纯函数 `observe.budget.budget_status(spent, budget, stop_reason, running)`：

| 级别 | 颜色 | 条件 |
|---|---|---|
| `OVER BUDGET`（超预算） | 红 | 已花费 > 预算；显示超出的美元数和百分比 |
| `BUDGET EXHAUSTED`（预算耗尽） | 橙 | 搜索以 `stop_reason == "budget_exhausted"` 结束（可以仍在预算内：闸门在下一次请求前触发） |
| `NEAR BUDGET`（接近预算） | 黄 | 仍在运行且已用至少 80% |

一次运行可以既超预算又耗尽，这时两个徽章都带。预算或花费未知的运行（旧结果）显示为「预算 —」，从不告警。界面上：

* 顶部固定横幅：各级别计数、总花费对预算合计、超出合计，以及每个被标记运行的可点击标签（首个 HTML 响应里就已渲染，没有 JavaScript 也能看到）；
* 历史列表：每次运行的预算、已花费、占比；被标记的运行带彩色徽章；筛选（全部 / 告警 / 超预算 / 耗尽 / 接近）和排序（时间 / 预算占比），可以把超预算的运行排到最前；
* 选中某次运行后，状态卡片和搜索面板重复显示徽章和精确金额。

`/api/budget` 以 JSON 返回同样的汇总，`/?run=p1%2Ffa_pro` 直接打开指定运行。数字取自每次运行自己的 `state.json` / `result.json` / `ledger.json`，所以这次改动之前启动的运行按它当时实际拥有的预算判断。

## 加预算与续跑

预算用完（`stop_reason == "budget_exhausted"`）或进程死掉（崩溃、编辑器窗口被关）的搜索，可以从它自己的目录继续；已经花掉的钱不会再花一次。两条路，同一个思路：

* **在线加预算**（搜索还在跑）：往 `<run_dir>/control.json` 原子写 `{"budget_usd": <新的总预算>}`。搜索在每次预算闸门前读它（一次 `stat`），把预算抬到这个数，永远不低于已经花掉的钱，并记录 `budget $a -> $b`。每次变化追加到 `<run_dir>/budget_events.json`（`requested` 是提出的人写的，`applied` 是搜索读到后写的，`resume` 是重新启动；每条有时间、旧新预算、金额、来源 `dashboard` / `cli`）。只有这次改动之后启动的进程才读这个文件，它的 `state.json` 里有 `"control_supported": true`；更早启动的进程不能在线加，等它停了再续跑。
* **续跑**已经停下的搜索：`uv run python -m taic.cli resume <run_dir> [--add-budget 美元 | --budget 新总额] [--dry-run]`。它从目录重建搜索状态，接着做没做完的轮次，再照常确认、收尾。覆盖任何文件之前，`result.json` / `confirm_report.json` 先留成 `*.before_topup_<n>.json`（旧 `state.json` 留成 `state.before_resume_<n>.json`），先前的结果不会丢。`--dry-run` 只在内存里重建、打印会从哪里继续（含各来源之间的核对），不写任何文件、不问模型、不用 GPU。因为别的原因停下的（收敛、`max_rounds`、GPU 挂死、模型出错）**不能**续跑，也没有强制开关；还活着的、或有活进程持着 `<run_dir>/resume.lock`（正在续跑）的目录会被拒绝。

续跑从哪里重建什么（细节见 `taic/resume.py`）：

| 状态 | 来源 | 是否精确 |
|---|---|---|
| 已花美元、token、请求列表 | `transcript.jsonl`（每次请求的 usage、花费、累计）；`ledger.json` / `state.json` 只用来核对 | 是（每个已经回复的请求） |
| 候选、成绩单、最好的内核、评测缓存 | `candidates/*/{kernel.metal,report.json}`；重放选择过程并与 `state.json` 的轨迹对照 | 是 |
| 试过的假设（`history`）、停滞轮数 | 转录里规划器的回复加每个分支的候选；并与每一份已记录的规划提示里的历史块逐条核对 | 是（模型请求*失败*的分支会写成「修复后仍无有效内核」） |
| 被打断的那一轮 | 方案取自转录（不再发规划请求）；已完成的分支保持完成；做到一半的分支从它记录下来的对话接着走：没有结果的工具调用在本地补答，下一个请求用原来该用的标签（`r4.p0.fix1#t1` ……） | 是（在途请求除外） |
| 系统提示 | 沿用转录里那一份，即使提示文件后来改过 | 是 |
| 设置 | `run_config.json`（新运行，不含凭据）；旧运行：预算和轮数上限取自 `state.json`，工具、画像技能、plan_mix、分支数、修复次数、初稿份数取自记录下来的提示和标签 | 旧运行没记 `patience` 和 `reasoning_effort`：用 `--patience` 给（默认取 `taic.toml`，即预设出现之前的运行用的值） |
| 前几轮的画像证据 | `profiles/r*_<种子>.{txt,json}` | 是；失败的画像不在列表里 |

不能恢复的：进程死掉时还在途的模型请求（服务商可能已计费，账本从没见过；续跑会列出它们），以及早先会话里每个请求的墙钟时间（那里 `t` 为 `null`）。试跑会把这两点列出来。

**看板。** 在预算耗尽 / 超预算 / 已中断的运行上（以及在跑且到了「接近预算」的运行上），运行的状态卡、顶部横幅的标签和搜索面板都有一个**加预算 Add budget**按钮。点开是一个小窗：当前预算、已花、目前最好加速比，`+$3 / +$5 / +$15` 和自定义金额，换算后的新总预算，以及「这会花真实的 LLM 费用，最多再花 $X」；确认之后页面照旧实时显示这次运行，并带「已加预算 TOPPED UP」徽章（它和固定预算的运行没有可比性）和预算历史（`$3.00 -> $8.00 (+$5.00, 04:35, dashboard)`）。不能继续的运行显示原因，不显示按钮。页面其余部分仍然只读。唯一的写入口（`POST /api/runs/<run>/budget`，请求体 `{"amount_usd": 5}` 或 `{"new_budget_usd": 8}`；`GET /api/runs/<run>/control` 说明现在能做什么）会花钱、会启动 GPU 工作，所以：服务只回应回环地址的 `Host`，浏览器的 `Origin` 必须是页面自己的，每个 POST 都要带页面里的每进程随机令牌（`X-Observe-Token`），金额必须为正且不超过 `OBSERVE_MAX_TOPUP_USD`（默认 25），运行目录必须落在 `runs/` 内（不许 `..`，不许符号链接逃出），每个运行同一时刻只处理一个请求。在线运行写 `control.json`；已停的运行启动一个独立会话的 `uv run python -m taic.cli resume ...`（日志在 `<run_dir>/resume_<n>.log`）。`OBSERVE_CONTROL_DRYRUN=1` 做完所有检查，把「会执行的命令」记进 `budget_events.json`（类型 `dryrun`），什么都不启动。

## 编译器基线（可选）：用 triton-msl 把 Triton 编译成 Metal

上游 Triton 没有 Metal 后端。每个任务的规格文本都是真正的 Triton，因此可以用社区后端 [triton-msl](https://pypi.org/project/triton-msl/)（alpha）来编译。它放在隔离环境里（`.venvs/triton-msl`，Python 3.14），只通过 `tcenv.triton_runner` 使用；主环境从不导入 Triton。

```bash
scripts/install_triton_msl.sh                          # 一次：隔离的 Python 3.14 环境（下载可续传，哈希已固定）
xcodebuild -downloadComponent MetalToolchain           # 一次：Apple 的 Metal 工具链，triton-msl 需要的可选 Xcode 组件
uv run python -m tcenv.cli triton-baseline gemm        # 或：all
```

来源：`triton-msl` 是固定了 sha256 的 PyPI wheel；Triton 3.7.0（上游提交 `4da2e268`）是 triton-msl 作者发布的**非官方预编译 wheel**，因为上游没有 macOS wheel（其 sha256 存在 `.venvs/triton-msl/TRITON_WHEEL.sha256`）。从源码构建 Triton 需要从 Triton CDN 下载 1.7 GB 的 LLVM，那是慢路径。

结果（相对 MLX 的时间，在同一进程里交错测量，扫描 `num_warps`，并检查正确性）缓存在 `runs/triton_baselines/`，并在评测报告里作为**估计**比较展示。「比 triton-msl 更快」比论文里「比 NVIDIA Triton 更快」弱得多。

## 目录

`tcenv/` 环境（契约、Metal 运行时、ABI 检查、sanitizer、测试、基准、剖析、工作进程、xctrace）—
`taic/` 智能体（LLM 客户端 + 账本、工具、提示词、编排器、技能）— `tasks/` 内核（Triton 规格、契约、参考实现、基线、朴素 MSL）—
`experiments/` — `tests/` — `runs/`（产物，已被 git 忽略）。
