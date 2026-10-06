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
uv run python -m taic.cli run softmax --budget 1.0     # 让大模型 lowering 一个内核（凭据：~/.pi/agent/models.json 或 OPENAI_*）
uv run pytest -q                                       # 环境、编排器（脚本化的大模型）、实验脚本
```

配置在 `taic.toml` 里；凭据从不写入仓库（从该文件所指的路径读取，或从 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 读取），并会在每一份记录里被打码。

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

## 实验（论文第 4 节）

```bash
uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16     # 图 4：加速比相对花费，12 个内核
uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5   # 表 1
uv run python -m experiments.semantic_tests --n-random 10                 # 第 4.1 节
uv run python -m tcenv.cli profile gemm best.metal                        # xctrace：spill、代码体积、热状态
```

怎么读这些数字：一次运行报告的加速比是**确认后**的加速比（胜者及其并列者用新种子、两倍重复次数重新测量）。套件表还会列出该任务手写朴素内核相对同一基线的加速比，因为有些基线偏弱（例如 Lion 和 BitDelta GEMM 没有融合的库实现）。朴素内核已经胜过基线时，收益里有一部分来自基线本身偏弱。

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
