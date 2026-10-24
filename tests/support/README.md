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
| `golden/<backend_id>/` | 按 backend 分目录的 golden 数据 | 数据 |

本目录不放业务测试（`tests/shared/test_structure.py` 会检查）。`hw_golden.py`
目前读取 `golden/metal/`，是 Metal-only 的 golden 工具；新增第二个 backend 时
再参数化或拆分。放置规则和并行说明见 `tests/README.md`。
