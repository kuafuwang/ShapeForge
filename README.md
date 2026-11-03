# ShapeForge

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
| P3-B 第二个 tiled GEMM 任务 | **blocked**：paper 通过，small 仍未稳定满足 fresh 1.10 门 |
| 清微 NPU | 仅离线模拟，NPU-P0 仍 blocked |

P3-B 不因为单次较好测量就标完成：只有稳定、fresh、逐 shard 的性能证据才算完成条件。

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

```bash
uv sync

# 查看标准任务合同
uv run python -m tcenv.cli tasks

# 让手写的简单内核走完整评测路径
uv run python -m tcenv.cli naive gemm

# 只读查看搜索摘要
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

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
- `docs/架构/跨硬件算子生成分工_9aef8d3a.plan.md`：OpSpec v2 与迁移架构；
- `docs/架构/构建执行与证据语义升级-plan.md`：构建/评测身份与发布证据；
- `docs/架构/执行运行时与设备扩展.md`：本地/远程执行与 evaluator 边界；
- `docs/security.md`：模型候选与 P2 安全边界；
- `docs/agent-skills.md`：agent skill 入口。

阶段验收材料位于：

- `docs/架构/跨硬件算子生成分工-阶段验收/`
- `docs/架构/shape-portfolio-阶段验收/`

这些是证据记录，不表示所有计划阶段已经完成。

## 历史来源

ShapeForge 起初是在 Apple M4 Max 上复现 *AI as a compiler: compiling Triton kernels without the Triton compiler*，早期路径是把 Triton 文本直接改写为 MSL。该历史保留在 Git 记录和可选 `triton-msl` 对照路径中。

Triton 已不再是项目边界。当前候选可以来自 Triton、手写 MSL、新算法或 ShapePlanner；当前主线是 OpSpec v2、候选自有实现、shape portfolio 与证据化激活。

## 安全与责任使用

Metal 内核可能让驱动显示器的 GPU 卡住。请使用有界 evaluator，不要直接手工启动不可信候选。真实 GPU hang 测试由 `TCENV_ALLOW_GPU_HANG_TESTS=1` 单独保护，不属于普通测试套件。
