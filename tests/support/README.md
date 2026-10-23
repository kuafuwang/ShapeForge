# Test support

这里放测试基础设施，不放业务测试：

- `fakes.py`：脚本化 LLM/响应；
- `handoff.py`：失败 AI handoff 插件；
- `run_logs.py`：xdist worker 日志插件；
- `fake_worker_*.py`：子进程 worker 替身；
- `orchestrator.py`：脚本化 LLM/evaluator 和 orchestrator 测试 helper；
- `resume.py`：resume/CLI 测试 helper；
- `triton_specs.py`：Triton spec 测试 helper；
- `llm.py`：脚本化 LLM 错误 helper；
- `hw_golden.py`：golden 读取/重录；
- `golden/<backend_id>/`：按 backend 分目录的 golden 数据。

新增 backend 专属 fixture 时放 `golden/<backend_id>/`；新增跨 backend
helper 才放本目录根下。
