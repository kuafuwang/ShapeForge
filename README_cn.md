# 在苹果显卡上，让模型把 Triton 写成 Metal

模型负责写小程序的文字。本仓库的程序负责编译、查安全、对答案、计时，并且只留下又安全、抽查算对、而且明显更快的那一版。模型看不见显卡，也不能自己宣布写对了或更快了。

能对外说的倍数，是搜索结束后换一批输入再测过、源码对得上、计时也不吵的那一个。搜索过程里看起来最快的那次，是另一个数。规则在 `experiments/common.py` 的 `conclusion_of_run`。

默认一次搜索大约 15 美元。两个模型轮流写：DeepSeek 的 `deepseek-flash`，然后是月之暗面的 `kimi-k3`。只设一家的密钥不够。`--preset base` 读的是 `taic.toml`，单次上限是 50 美元。`taic run` 和 `taic resume` 会花钱，也会占用这台同时画屏幕的显卡。只看摘要用 `uv run python -m taic.cli brief`，不花钱、不占显卡。

本文其余部分与 [README.md](README.md) 对齐。命令、路径和配置名保持英文。下面这张表是和论文的对照，不是上手步骤。

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
| GPT-6 Astra、美元预算 | 任意 OpenAI 兼容端点。默认两个席位轮流：`deepseek-flash` 与 `kimi-k3`。账本按缓存命中分开计价 |

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
* 每次评测都在工作进程里运行，该进程在每次 GPU 启动前后打心跳；一条 command buffer 在途过久，父进程就 `SIGKILL`：单核 2 秒（着色器校验 4 秒）；流水线按段数乘 2 秒、封顶 8 秒，校验再乘 2、最多 16 秒；并强制执行总超时；
* 第一次评测超时后，编排器会停止**整次运行**（`max_hangs = 1`）；
* 真正的挂起测试默认跳过，除非设置 `TCENV_ALLOW_GPU_HANG_TESTS=1`；监督逻辑改由假工作进程覆盖。

## 测量卫生

并发任务会扭曲计时：套件还在跑的时候测一次 triton-msl，结果会慢 2–4 倍；一个变异系数（cv）为 16% 的 GEMM 候选（安静运行时为 0.5–1.5%）曾挤掉更好的内核。因此：

* 本项目里的每一次计时（候选基准、基线、triton-msl、标定、xctrace 录制）都独占一把机器级锁；评测器的其他 GPU 工作则以共享方式持有同一把锁。这样不同实验可以同时跑，测量时段却不会重叠（`tcenv/gpulock.py`，对 `~/.cache/ai_lowerling/gpu.lock` 做 `flock`，可用 `TCENV_GPU_LOCK_FILE` 覆盖）。等锁的时间不计入评测超时。本项目之外的程序不受这把锁约束；
* 候选或库基线的单次重复变异系数超过 5% 会重测（最多 3 次，保留最安静的一次，其余由 `benchmark.noisy` 标记）。`candidate_serial` 不触发重测；
* 候选要替换当前最优，必须同时快过 `min_improvement`，并且快过两次加速比的 `noise_z` 个标准误；
* 结束时，最优者以及在噪声范围内与它并列的最多 `confirm_top - 1` 个候选会被重新测量；最快的成为结果（`result.json` 的 `confirmations`）。

## 预算

论文给每次搜索设了固定的美元上限（arXiv 2609.36800v1，`docs/2609.36800v1.pdf`）：第 4.3 节和表 1，「卷积、矩阵乘、FlashAttention 每次运行最多 $15，较简单的逐元素 SwiGLU 内核 $1」；第 4.4 节和图 5，十个已发表内核「在 $15 预算下」。搜索一直重复到预算耗尽（第 3.3 节）；预算在每次 LLM 请求之前检查，所以已经发出的请求可以略微超出上限。本仓库照此办理，**默认预算是每次运行 $15**（`SearchConfig`、`taic.cli run`、`run_suite`、`repeatability`、`p1_runs`）。图 4 的 12 内核网格没有写明上限，同样用 $15；重复性实验（表 1）像论文一样给 SwiGLU $1。

* **配置在哪。** `taic.paper.toml` 是叠加在 `taic.toml` 之上的预设（设了 `budget_usd = 15` 和 `max_rounds = 100`：$15 时结束运行的是预算而不是轮数，原来的 12 轮上限会在花掉约三分之一的钱时就停下）。`--preset base` 或 `TAIC_PRESET=base` 只读 `taic.toml`；`--config FILE` 原样使用。预设把 `patience`（连续 N 轮没有新最优就停）从 6 提到 12。`taic.toml` 自己是 6，不是 4。短暂的平台期不会截断 $15 的运行，真正收敛的运行仍会提前结束，所以不再进步的内核花不到 $15。
* **覆盖。** 到处都有 `--budget USD`，`run_suite` 和 `repeatability` 还有 `--budget-scale F`（例如 `0.2` 做冒烟）。语义测试（每次 $0.15）、`counter_teaching`（$1）和单元测试特意保留小预算。
* **最坏情况。** 若每次运行都花满预算：`run_suite --tasks all --dtypes fp16,bf16` 是 24 次运行、$360（默认只写 `--dtypes` 时是 fp16，12 次、$180）；`repeatability` 是 5 次 x（$15 + $15 + $1）= $155（加上 `flash_attention` 为 $230）；`p1_runs` 把字典里 12 条都启动是 $180。这三个驱动都有可选的 `--max-total-usd USD`（默认关闭）：整次调用放不下的运行会被跳过。磁盘上已经有花到约 $46 的单次搜索（`runs/conv2d-two-models-20261019`，上限 $50）。早先「已结束运行最多 $3.12」只对当时的 `runs/p1/fa_flash` 成立。
* **更早的结果不可比。** 它们用的是更小的预算，看板逐运行显示（每个 `result.json` 也记录 `budget_usd`）：`runs/suite_full` 多数内核 $0.5–$1.2，其中 `gemm_fp16` 的预算字段已是 $15；`runs/repeat_*` 每次试验 $3，`runs/p1` 多数 $3（后来启动的 `*_paper` 运行是 $15），`runs/semantic_main` $0.15，`runs/pilot_suite_mini_old_protocol` $1.25–$1.5。在那里以 `budget_exhausted` 结束的运行，在论文预算下本会继续搜索。

## 实验（论文第 4 节）

```bash
uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16     # 图 4：加速比相对花费，12 个内核（每次 $15；冒烟用 --budget-scale 0.2）
uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5   # 表 1（$15，swiglu $1）
uv run python -m experiments.semantic_tests --n-random 10                 # 第 4.1 节：10 道随机核再加 4 道误导核，每道 $0.15
uv run python -m tcenv.cli profile gemm best.metal                        # xctrace：spill、代码体积、热状态
```

怎么读这些数字：一次运行报告的加速比是**确认后**的加速比（胜者及其并列者用两倍重复次数重新测量；有 `run_seed` 的运行用 confirm 阶段派生的种子，不是另抽随机数）。套件表还会列出该任务手写朴素内核相对同一基线的加速比，因为有些基线偏弱（例如 Lion 和 BitDelta GEMM 没有融合的库实现）。朴素内核已经胜过基线时，收益里有一部分来自基线本身偏弱。

## 硬件特性反馈（可选，默认关闭）

`search.hw_feedback = true`（`taic run ... --hw-feedback`、`taic resume ... --hw-feedback`，或驱动脚本的搜索选项里写 `hw_feedback=True`，和 `profile_poor` 一样）会在这一场 `TASK.md` 的证据段里，
profiler 证据所在的位置，附上一小段（至多 12 行、约 150 词），并把 `taic/skills/hardware_feedback.md` 作为 Skill 2c 写进存盘的系统提示。`pi` 读的是任务书，不是那份提示。这一段由 `taic/hwfeedback.py` 根据 `docs/hardware_features.md` 的静态检测器
（`taic/hwfeatures.py`：源码、编译出的 AIR、以及这颗候选自己测得的 roofline）生成，内容是：

* 内核已经用了哪些 Apple GPU 特性（矩阵单元及其累加器类型和 fragment 个数、threadgroup 暂存与 padding、双缓冲或寄存器预取、device 读写的宽度、SIMD 归约与 shuffle、online softmax 等），以及离 roofline 多远；
* 这一类任务本应有、却没找到的特性，至多三条，每条一句具体提示（例如「没有 simdgroup_matrix：矩阵乘在标量 FMA 上跑；roofline 显示是下界的 3.0 倍」），并且总带一句说明：这些是静态分析给的提示，不是要求，以实测时间为准；
* 已经在 roofline 的约 90% 以上的内核不列任何缺失项（超过 roofline 的部分硬件特性帮不上忙；逐元素任务就处于这种状态，标量读写已经接近带宽：`docs/hardware_features.md` 第 6 节）；
  内核有 32 个以上累加器 fragment 时给出警告（参考 GEMM 慢了 12 到 26 倍），处于寄存器压力上限时也一样。

为什么默认关闭：这是实验，目前没有证据表明它有用。关闭时，系统提示、用户消息和反馈与打开之前的形状对照；`tests/test_hw_feedback.py` 用 `tests/golden/hw_off.json` 核对 176 份系统提示、110 条用户消息、12 段反馈。这份黄金文件在 2026-10-20 重录过，里面的脚本化搜索不止早先写的四次。`taic.toml` 和 `taic.paper.toml` 没有提到它。它记在 `run_config.json` 里，`taic resume` 会保留。
这段分析不会弄坏或拖慢搜索：超时（20 秒）、失败或没有 Metal 工具链时，直接去掉这一段并记一条警告。A/B 实验：`experiments.p1_runs` 的 `fa_flash_hw` 就是只改了这个开关的 `fa_flash`（n=1，随机搜索，只能算轶事）；这次运行于 2026-10-07 06:08 启动，因检测器修正中途停过一次并用 `taic resume` 接着跑（第 1 轮看到的是修正前的提示块，见 `docs/hardware_features.md` 第 9 节）。结果已经落盘：`runs/p1/fa_flash_hw` 的 `confirmed_speedup` 约 0.622，花了约 $2.89，8 轮后因连续 4 轮没有新种子而停。对照 `fa_flash` 的确认倍数约 0.735、$3.12、8 轮、预算耗尽。两次都是 $3 预算下的单次搜索，而且都没有绑上今天发布闸门要的源码指纹，不能当成「硬件提示有没有用」的结论。

硬件特性反馈打开时，如果某个基线已经比 roofline 下界快 5% 以上，这段话不会再说「已经到顶」。下界按直接算法的运算量来算，基线比它还快，说明基线用了更少的运算（测到的例子是 conv2d：torch-mps 777 微秒，直接卷积的下界 932 微秒，大约是实测矩阵峰值的 1.2 倍）。可达目标是那个基线的时间。

## 搜索开关（默认打开）

每一个默认都开着。关掉用 `--no-<旗标>`，例如 `--no-stop-at-roofline`，其余同名。`truncation_retry` 在模型配置上，其余在搜索配置上。它们写进 `run_config.json`。`taic resume` 保留已经记下的值；这次开关出现之前启动、目录里没有记下的运行，续跑时按默认处理，也就是打开。下面这六个 `--no-*` 只关掉这张表。计划账本是另一只开关，默认开着，要 `--no-plan-ledger` 才关。

| 开关 | 改变什么（默认开，`--no-` 关掉） |
| --- | --- |
| `--truncation-retry` | 字段仍默认开，强度默认仍是 `low`。当前一场不会再问。再问一次的代码在不再调用的 `_develop` 里 |
| `--stop-at-roofline` | 种子已经落在自己 roofline 的噪声门限以内，且没有基线比这个下界更快时，停止搜索（`stop_reason=at_roofline`） |
| `--abi-guard` | 字段仍默认开。当前一场不会因为重复的接口错误停掉一支。停分支的代码在不再调用的 `_develop` 里 |
| `--target-note` | 某个基线比 roofline 下界快 5% 以上时，任务书写明这个下界不成立，目标是该基线的时间 |
| `--rich-history` | 失败那一轮的历史行写明失败种类（`abi:`、`incorrect:`、`no code:`），不再只有一句笼统的话 |
| `--status-lines` | 任务书写明这次至少要快百分之几才算赢，以及还剩几轮、多少耐心和多少预算 |

`result.json` 和 `state.json` 另外记下 `search_seed_label`（搜索正在改的那颗内核）和 `confirm_winner_label`（换种子重测后留下的那颗）。续跑从搜索种子接着改。

另外三只默认关着的开关：`--diversity-archive`（选择函数还在，当前循环不调用，`parent_label` 保持空）、`--fast-bench`（搜索阶段少测几条基线；确认测量仍测全）、`--experience-index`（指向一份现成 JSON 才在任务书里附最多 3 行旧提示，默认空）。计划账本默认开：看板可按它重算计划结局。当前任务书只附最近 8 行普通历史。

套件、重复性实验和 `p1_runs` 的 `--max-total-usd` 按磁盘上预留的钱来算：还在跑的运行算它的当前预算（看板上加过的、旁边单独续跑的都算进去），已经结束的只算实际花掉的。上限写在实验目录的 `cap.json` 里，看板拒绝一笔会冲破这个上限的加预算。

## 看板与预算告警

`uv run python -m observe` 启动本机看板（默认 `http://127.0.0.1:8765`；改了代码要重启，页面在启动时读入）。页面有实验、臂对照（发布加速比、`fast_p`、每美元增益）、分页的运行列表、单次运行，以及花费账本。看板会写磁盘：加预算、重置预算、停止进程、改下次搜索的默认模型、删除记录。从旧目录名归到一组的运行标成推断，不下胜负。读接口是 `/api/v2`。它把每次运行的预算和已花费并排显示，并对越过预算的运行告警。规则是一个纯函数 `observe.budget.budget_status(spent, budget, stop_reason, running)`：

| 级别 | 颜色 | 条件 |
|---|---|---|
| `OVER BUDGET`（超预算） | 红 | 已花费 > 预算；显示超出的美元数和百分比 |
| `BUDGET EXHAUSTED`（预算耗尽） | 橙 | 搜索以 `stop_reason == "budget_exhausted"` 结束（可以仍在预算内：闸门在下一次请求前触发） |
| `NEAR BUDGET`（接近预算） | 黄 | 仍在运行且已用至少 80% |

一次运行可以既超预算又耗尽，这时两个徽章都带。预算或花费未知的运行（旧结果）显示为「预算 —」，从不告警。界面上：

* 顶部固定横幅：各级别计数、总花费对预算合计、超出合计，以及每个被标记运行的可点击标签（首个 HTML 响应里就已渲染，没有 JavaScript 也能看到）；
* 运行名单的列是运行、题目、发布、加速比、绑定、状态。筛选是题目、dtype、模型和状态。以前的筛选胶囊和排序胶囊没有画出来；
* 选中某次运行后，状态卡片和搜索面板重复显示徽章和精确金额。

`/api/budget` 以 JSON 返回同样的汇总，`/?run=p1%2Ffa_pro` 直接打开指定运行。数字取自每次运行自己的 `state.json` / `result.json` / `ledger.json`，所以这次改动之前启动的运行按它当时实际拥有的预算判断。

## 加预算与续跑

预算用完（`stop_reason == "budget_exhausted"`）或进程死掉（崩溃、编辑器窗口被关）的搜索，可以从它自己的目录继续；已经花掉的钱不会再花一次。两条路，同一个思路：

* **在线加预算**（搜索还在跑）：往 `<run_dir>/control.json` 原子写 `{"budget_usd": <新的总预算>}`。搜索在每次预算闸门前读它（一次 `stat`），把预算抬到这个数，永远不低于已经花掉的钱，并记录 `budget $a -> $b`。每次变化追加到 `<run_dir>/budget_events.json`（`requested` 是提出的人写的，`applied` 是搜索读到后写的，`resume` 是重新启动；每条有时间、旧新预算、金额、来源 `dashboard` / `cli`）。只有这次改动之后启动的进程才读这个文件，它的 `state.json` 里有 `"control_supported": true`；更早启动的进程不能在线加，等它停了再续跑。
* **续跑**已经停下的搜索：`uv run python -m taic.cli resume <run_dir> [--add-budget 美元 | --budget 新总额] [--dry-run]`。只有 `run_config.json` 里 `loop_protocol` 为 `flame_chase_1` 的目录能续，其它退出码 5。它从目录重建搜索状态，接着做没做完的轮次，再照常确认、收尾。已经有候选的轮次不重做。没有候选的轮次新开一场。覆盖任何文件之前，`result.json` / `confirm_report.json` 先留成 `*.before_topup_<n>.json`（旧 `state.json` 留成 `state.before_resume_<n>.json`），先前的结果不会丢。`--dry-run` 只在内存里重建、打印会从哪里继续（含各来源之间的核对），不写任何文件、不问模型、不用 GPU。因为收敛或轮数用完而停下、并且已经有合格稿的，可以加 `--lift-count-limits` 再续。图形芯片挂死和模型接口出错仍然不能续。还活着的、或有活进程持着 `<run_dir>/resume.lock`（正在续跑）的目录会被拒绝。看板「重置预算」会给每个还能继续的搜索加 15 美元，并把轮数和耐心抬到 10 万，续跑命令里带这个开关。

续跑从哪里重建什么（细节见 `taic/resume.py`）：

| 状态 | 来源 | 是否精确 |
|---|---|---|
| 已花美元 | 转录里 `agent` 事件的 `cost_usd` 累加。还原出的缓存命中、未命中和输出 token 是 0 | 美元是；token 拆分不是 |
| 候选、成绩单、最好的内核、评测缓存 | `candidates/*/{kernel.metal,report.json}`；重放选择过程并与 `state.json` 的轨迹对照 | 是 |
| 试过的假设（`history`）、停滞轮数 | `state.json` | 当时存下的那些行 |
| 被打断的那一轮 | 已经有候选的轮次算做完。没有候选的轮次新开一场并再计费。`partial` 固定为空 | 只对 `flame_chase_1` |
| 系统提示 | 有 `system_prompt.txt` 就读它。下一场仍由当前代码现写 `TASK.md` | 读的是这个文件，不是转录原文 |
| 设置 | `run_config.json`（新运行，不含凭据）；旧运行：预算和轮数上限取自 `state.json`，工具、画像技能、plan_mix、分支数、修复次数、初稿份数取自记录下来的提示和标签 | 旧运行没记 `patience` 和 `reasoning_effort`：用 `--patience` 给（默认取 `taic.toml`，即预设出现之前的运行用的值） |
| 前几轮的画像证据 | `profiles/r*_<种子>.{txt,json}` | 是；失败的画像不在列表里 |

不能恢复的：进程死掉时还在途的模型请求（服务商可能已计费，账本从没见过；续跑会列出它们），以及早先会话里每个请求的墙钟时间（那里 `t` 为 `null`）。试跑会把这两点列出来。

**看板。** 在预算耗尽 / 超预算 / 已中断的运行上（以及在跑且到了「接近预算」的运行上），运行的状态卡、顶部横幅的标签和搜索面板都有一个**加预算 Add budget**按钮。小窗是一个金额框，默认 1。同一页还能重置、停止全部搜索、改下次两个委员会的模型，以及带理由删除已结束的记录。确认之后页面照旧实时显示这次运行和预算历史。不能继续的运行显示原因，不显示按钮。加预算的接口是 `POST /api/runs/<run>/budget`，请求体 `{"amount_usd": 5}` 或 `{"new_budget_usd": 8}`；`GET /api/runs/<run>/control` 说明现在能做什么。这个接口会花钱、会启动 GPU 工作，所以：服务只回应回环地址的 `Host`，只有请求带了 `Origin` 才核同源，每个 POST 都要带页面里的每进程随机令牌（`X-Observe-Token`），金额必须为正且不超过 `OBSERVE_MAX_TOPUP_USD`（默认 25），运行目录必须落在 `runs/` 内（不许 `..`，不许符号链接逃出），每个运行同一时刻只处理一个请求。在线运行写 `control.json`；已停的运行启动一个独立会话的 `uv run python -m taic.cli resume ...`（日志在 `<run_dir>/resume_<n>.log`）。`OBSERVE_CONTROL_DRYRUN=1` 做完所有检查，把「会执行的命令」记进 `budget_events.json`（类型 `dryrun`），什么都不启动。

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
