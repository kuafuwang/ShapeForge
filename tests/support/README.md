# Test support

## 这篇讲什么

- 这里放测试基础设施（helper、fake、pytest 插件、golden 数据的读盘工具），
  不放业务测试。
- 新增 backend 专属 fixture 时放 `golden/<backend_id>/`；只有新增跨 backend
  的 helper 才放本目录根下。

## 模块清单

| 模块 | 作用 | 类别 |
| --- | --- | --- |
| `fakes.py` | 脚本化 LLM/响应 | fake |
| `fake_worker_*.py` | 子进程 worker 替身 | fake |
| `orchestrator.py` | 脚本化 LLM/evaluator 和 orchestrator 测试 helper | helper |
| `resume.py` | resume/CLI 测试 helper | helper |
| `triton_specs.py` | Triton spec 测试 helper | helper |
| `llm.py` | 脚本化 LLM 错误 helper | helper |
| `dashboard.py` | 共享的 dashboard fixture/pytest 插件 | pytest 插件 |
| `handoff.py` | 失败 AI handoff 插件 | pytest 插件 |
| `run_logs.py` | xdist worker 日志插件 | pytest 插件 |
| `hw_golden.py` | golden 读取/重录 | Metal-only 工具 |
| `shape_portfolio.py` | shape portfolio P0 的固定有限题/源/几何/证据 fixture | helper |
| `remote_contract.py` | 对外 evaluator 合约 golden 的路径与只读 loader | helper |
| `boundaries.py` | 当前迁移边界的唯一派生处：G0 冻结边界 + 各批次自己的 `boundary.json` | helper |
| `v1_specimen.py` | 已迁移样本题（lion/conv2d）的任务形状夹具，供 v1 机制测试（含真实合同文本） | helper |
| `varlen_probe.py` | 用 shipped `reference.py` 写一个只换 `sizes` 的探针任务目录（varlen case 覆盖） | helper |
| `npu_sim/varlen_worklist.py` | V2 的 NPU 语义模拟：work list/core 认领/UB-L1 预算/handoff 事件图/对抗调度/DMA 对齐；只产 `simulated=true/publication_allowed=false` | helper |
| `hostprogram_fixtures.py` | V3 的 host+device msl-bundle envelope（`host/tiling.c` 计划、geometry、可替换 planner 源码） | helper |
| `hostprogram_oracle.py` | V3 的独立 Python planner oracle（range-cut/LPT/覆盖检查）；只供测试差分，不进 candidate/timed path | helper |
| `golden/shared/hostprogram/kda_planner.c` | V3 候选侧受限 C99 KDA planner（mode 1 均衡切分、mode 2 range-cut + heap LPT） | 数据 |
| `golden/shared/` | backend 无关的冻结文档夹具（如 B1 前的 rope v1 opspec，供 v1 机制测试） | 数据 |
| `remote_pki.py` | 用 openssl 生成一次性 CA/server/client 证书的测试 PKI | helper |
| `process_state.py` | 不依赖 `ps` 的进程状态/回收探针（围栏环境） | helper |
| `remotegrpc/` | 独立 Go gRPC conformance client（descriptor 动态消息、digest/验签） | 跨语言测试工具 |
| `npu_sim/grpc_service_entry.py` | 仅测试用的 gRPC composition root（fake worker） | test-only entry |
| `npu_sim/grpc_sim_runner.py` | 2D2 本机 Docker gRPC/TLS 验收与报告生成 | integration runner |
| `npu_sim/tcp_proxy.py` | gRPC/TLS 验收的 reset/half-open/slow TCP 故障注入 | fault helper |
| `remotecontract/` | 独立 Go digest/receipt/HMAC/timeout golden verifier | 跨语言测试工具 |
| `backend_neutral_driver.py` | fresh-interpreter 的 B0.5 Metal import 阻断与 fake 评测驱动 | support helper |
| `golden/<backend_id>/` | 按 backend 分目录的 golden 数据 | 数据 |
| `golden/remote/` | backend 无关的对外协议 golden；不是新 backend | 数据 |
| `docker/Dockerfile.remote-grpc-sim` | 2D2 的测试专用 fake-backend gRPC 镜像 | test image |

本目录不放业务测试（`tests/shared/test_structure.py` 会检查）。`hw_golden.py`
目前读取 `golden/metal/`，是 Metal-only 的 golden 工具；新增第二个 backend 时
再参数化或拆分。放置规则和并行说明见 `tests/README.md`。
