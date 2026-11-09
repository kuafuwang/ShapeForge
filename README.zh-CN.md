# ShapeForge

[English](README.md) | [简体中文](README.zh-CN.md)

**Generate specialized kernels. Prove every shard.**  
**生成专用内核，用证据证明每一个分片。**

ShapeForge 是一个证据优先的算子生成、评测与精确分派系统。候选可以由 LLM 或人编写，不再局限于 Triton 直译，也不受封闭式 pattern 库限制。

当前生产目标是 **Apple GPU / Metal Shading Language**。清微 NPU 目前只有离线协议与数据模拟，不宣称真机 correctness/performance，也不解除 NPU-P0 阻断。

## 项目定位

- **任务合同**：通过标准 OpSpec v2 目录描述语义、输入输出、容差和已登记 baselines；
- **候选拥有实现**：候选选择算法、融合、Tiling、Host Tiling 与 ExecutionGeometry；
- **Shape portfolio**：候选 ShapePlanner 声明 coverage buckets，多个稳定 variant 共存，exact one-of-N dispatch 至多选择一个实现；
- **独立 fallback**：未覆盖输入走独立验证过的通用实现，不能把 fallback 的 correctness 或时间记成 variant 的功劳；
- **未见分片验证**：required boundary、tail、resource instances 不进入候选调优视图；
- **证据化激活与发布**：CanonicalBundle、构建/评测身份、fresh confirmation 和唯一发布谓词共同决定能否激活、恢复与引用性能数字。

Harness 拥有任务合同、ABI、安全、测量边界与证据；不为候选规定封闭算法集合。

## 当前状态

| 范围 | 状态 |
| --- | --- |
| R0–R6 构建与证据身份 | 已完成 |
| G0、A1–A6 OpSpec v2 迁移 | 已完成 |
| Shape portfolio P0–P2 | 已完成 |
| P3-A 性能两本账 | 已完成 |
| Varlen V0–V6 与固定长度 KDA 交付 | varlen Metal 验收完成；固定 KDA 标准任务与复杂 bundle 通过离线/CPU 检查，真实 Metal bundle 与付费搜索未执行 |
| P3-B 第二个 tiled GEMM 任务 | **blocked**：paper 通过，small 仍未稳定满足 fresh 1.10 门 |
| 清微 NPU | 仅离线模拟，NPU-P0 仍 blocked |

P3-B 不因为单次较好测量就标完成：只有稳定、fresh、逐 shard 的性能证据才算完成条件。

## 多 DSL Track

Track 是交付、构建与执行路线，不是 DSL，也不是中央 IR。共享 Metal runtime 负责安全、缓冲区、dispatch 审计、计时和证据；各 Track 自己拥有 frontend 与执行计划。

| Track | 当前证据 |
| --- | --- |
| `msl-single` / `msl-bundle` | 既有 MSL 路线，`affine_ramp` fresh confirmation 通过 |
| `tilelang-pipeline/1` | `interface_only` source-only replay，不可选择 |
| `tilelang-pipeline/2` | 固定人工 TileLang Metal 机制和 correctness-only 生成/replay 已完成 |
| `tilelang-pipeline/3` | 固定 ReLU 登记、activation/resume 和 `affine_ramp` exploration winner；`track_explore_reviewer_done=true` |

r15 覆盖三条路线，各自拥有独立 run_dir 和 fresh confirmation。TileLang `/3` 的 latency ratio 最低（0.074707），但这只是 **exploration**：不是同一 fresh 输入、不是跨 Track runtime portfolio，也不是 speedup 发布。MT6 已按独立复核后的 MSL + TileLang 范围完成（`full_mt6_done=true`，`additional_dsl_not_required=true`）。`publication_allowed=false`。MT7 最终独立终审已完成，但这不开放 speedup publication。证据见 `docs/架构/多DSL-Track-阶段验收/MT6/`；observe 提供只读 `/api/v2/track-explore` 入口。

## 候选与 Harness 的分工

```text
候选（人或 LLM）
  算法 / 融合 / coverage plan / Tiling / Host Tiling / ExecutionGeometry
                              │
                              ▼
Harness
  OpSpec + ABI + pack 闭包 + 安全 + 有界执行
                              │
                              ▼
评测与证据
  correctness + dispatch parity + fresh timing + canonical identity
                              │
                              ▼
激活与发布
  exact dispatch + fallback + confirmation + revocation/resume
```

候选可以使用 persistent kernel、多阶段状态机和公开的原生矩阵/同步能力。任意宿主 include、未声明资源和沙箱逃逸仍会被拒绝。runaway 由外部看门狗处理，不靠限制候选循环风格来保证安全。

## 快速开始

环境要求：Apple Silicon macOS、Xcode/Metal 工具链、`uv`，以及 `pyproject.toml` 声明的 Python 依赖。

缺东西时，先跑一条命令自检：`uv run python scripts/doctor.py` 会逐项检查并只给出每条问题的下一步。

```bash
uv sync

# 一份自检：工具链、GPU、任务、运行目录、看板
uv run python scripts/doctor.py

# 查看标准任务合同
uv run python -m tcenv.cli tasks

# 让手写的简单内核走完整评测路径
uv run python -m tcenv.cli naive gemm

# 只读查看搜索摘要
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

### 跑对了长什么样

| 命令 | 该看到的输出 |
| --- | --- |
| `scripts/doctor.py` | 全部通过时是十一行 `[PASS]`，再加一行汇总；只有 WARN 或 FAIL 才有 `Next:`；退出码 0 |
| `tcenv.cli tasks` | 每个 **v2** 任务一行，带 `dtype=`、`size=`、`inputs=`、`flops=`（`--legacy` 是已退役的 v1 登记，现在不打印任务；v1 夹具在 `tests/support/golden/shared/v1-track/`） |
| `tcenv.cli naive gemm` | 先给正确性结论，再给微秒数和加速比 |
| `taic.cli brief` | 搜索最优与确认加速比分开列出 |

### 报错怎么办

| 报错 | 原因 | 怎么修 |
| --- | --- | --- |
| `[Errno 48] error while attempting to bind on address ('127.0.0.1', 8765)` | 已经有一个看板占着这个端口 | 换端口 `OBSERVE_PORT=8766 uv run python -m observe`，或停掉旧的：`lsof -nP -tiTCP:8765 -sTCP:LISTEN \| xargs kill` |
| `ModuleNotFoundError: No module named 'tcenv'` | 没在项目环境里跑 | 先 `uv sync`，命令前加 `uv run` |
| 不认识的 `code=` | 任务合同或证据平面的 typed 失败 | 查 [docs/失败码.md](docs/失败码.md)：含义、常见原因、下一步 |

### 测试

先运行不使用 GPU、模型和长检查的 CPU 安全套件：

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

只有明确允许真实 GPU 使用时才运行：

```bash
uv run pytest -q -m gpu
```

调试单个失败时，先读 `logs/pytest/triage/triage_summary.md` 及其指向的 Markdown/JSON handoff，再串行复现：

```bash
uv run pytest -q -n 0 <nodeid>
```

测试目录、marker、golden 和并行规则见 `tests/README.md`。

### 付费模型搜索与 GPU 使用

`taic run` 和 `taic resume` 会花钱，也会在这块同时驱动显示器的 GPU 上执行内核。读取证据和运行 CPU 套件不需要它们。

```bash
# 仅为示例：会花钱并使用 GPU
uv run python -m taic.cli run softmax --budget 1.0
```

写内核通过本机 `hmz` 启动器和 `pi`，见 `docs/架构/hmz.md`。凭据从环境变量或配置的凭据文件读取，绝不提交到仓库。

## 证据与性能口径

两个数字不能混用：

- **Search best**：搜索期间看到的最快观察值，不可直接发布；
- **Published speedup**：搜索后重新测量，源码与评测身份匹配、计时安静，并通过唯一发布门的结果。

正确性只认本仓库的 `tolerances + scaled_error`，并包括 `read_write` 终值。fallback 的 correctness 和时间不计入 variant 的 correctness 或 speedup。

Shape portfolio 使用两本账：

- `specialized_hit_speedup`：只计算由生成 variant 处理的 shards；
- `effective_end_to_end_speedup`：计算全部实际工作，包括真实 fallback 路径。

每个 shard 独立满足冻结的回归政策。geomean 只能描述，不能掩盖局部退化。原始 observation、JSON 字段或摘要本身都不能证明真实硬件执行；激活还需要独立 expected 重建和 fresh evidence。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `tasks/` | OpSpec v2 标准目录：`opspec.json`、`reference.py`、baselines 和可选 display/reference 元数据 |
| `tcenv/` | 任务合同、Track/CanonicalBundle、评测、Metal runtime 与后端中立证据 |
| `tcenv/portfolio/` | Semantic Domain、候选 portfolio 与 exact dispatch |
| `tcenv/portfolio_evidence/` | 确定性证据投影与严格比较 |
| `tcenv/portfolio_validation/` | freeze、tune/unseen plan 和候选视图 |
| `tcenv/portfolio_performance/` | 逐 shard 判定与两本性能账 |
| `taic/` | 搜索、确认、激活、持久化与 portfolio-aware resume |
| `backends/` | backend 元数据、skills 和 wiki |
| `targets/` | 芯片 profile、测量和信任边界 |
| `tests/` | `shared/`、`backends/<id>/` 和 support/golden 套件，规则见 `tests/README.md` |
| `docs/` | 架构、实施计划、安全边界与阶段验收证据 |
| `kernels/` | 已确认的内核示例（若存在） |

历史 v1 任务只作为测试夹具保留。旧的逐题 `task.py`、`track.json`、`kernel_triton.py`、逐题 LoweringProfile 和生成任务中央登记表不是当前 authoring path。

## 文档导航

- `docs/架构说明.md`：进程与 runtime 总览；
- `docs/架构/shape-portfolio实施计划.md`：portfolio 权威计划与 P0–P4 状态；
- `docs/架构/v2协议参考.md`：OpSpec v2 题面与候选交付；
- `docs/架构/构建与评测身份.md`：构建/评测身份与发布证据；
- `docs/架构/执行运行时与设备扩展.md`：本地/远程执行与 evaluator 边界；
- `docs/security.md`：模型候选与 P2 安全边界；
- `docs/agent-skills.md`：agent skill 入口。

阶段验收材料位于：

- `docs/架构/跨硬件算子生成分工-阶段验收/`
- `docs/架构/shape-portfolio-阶段验收/`

这些是证据记录，不表示所有计划阶段已经完成。

### 本地看板

一键启动或安全重启：

```bash
scripts/observe.sh
```

脚本先探测 `OBSERVE_PORT`（默认 8765）上的旧实例，停止它并等待端口释放，再启动新实例，最后做 HTTP 健康检查。日志和 PID 状态位于 `${TMPDIR:-/tmp}/shapeforge-observe-$UID/`，可通过 `OBSERVE_PORT` 换端口。

## CI

当前没有接线：仓库里已不再保留 workflow 定义，push 不会触发任何检查。改为在本地跑同样的两条路径：

- **CPU 安全套件**——确定性 smoke 与 portfolio 测试，不调用付费模型：
  `uv run pytest -q -m 'not gpu and not llm and not slow'`。
- **Metal GPU**——仅手动，在本机空闲的 Apple Silicon 上执行，且无付费模型凭据：
  `uv run pytest -q -n 0 -m gpu`。

无论谁来跑，都不允许使用 `continue-on-error`，失败和 `logs/pytest/triage/` 证据都要保留。以后重新接入 CI 时，
两条路径必须保持分离，GPU 检查不得跑在共享 runner 上。

## AI 协助贡献

其他参与者和 AI 应先读 `AGENTS.md`，它是权威项目上下文与安全边界。`CONTRIBUTING.md` 说明人类/AI 协作流程，`.agents/skills/shapeforge/SKILL.md` 提供简短的上手 skill。通过这些文件可以快速理解候选自由、证据不变量、当前状态、测试方式和授权边界，不必再从旧的 Triton 历史推断项目。

## 历史来源

ShapeForge 起初是在 Apple M4 Max 上复现 *AI as a compiler: compiling Triton kernels without the Triton compiler*，早期路径是把 Triton 文本直接改写为 MSL。该历史保留在 Git 记录和可选 `triton-msl` 对照路径中。

Triton 已不再是项目边界。当前候选可以来自 Triton、手写 MSL、新算法或 ShapePlanner；当前主线是 OpSpec v2、候选自有实现、shape portfolio 与证据化激活。

## 安全与责任使用

Metal 内核可能让驱动显示器的 GPU 卡住。请使用有界 evaluator，不要直接手工启动不可信候选。真实 GPU hang 测试由 `TCENV_ALLOW_GPU_HANG_TESTS=1` 单独保护，不属于普通测试套件。
