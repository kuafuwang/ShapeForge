# Metal tests

这里放只对 Metal backend 有意义的测试。当前第一批包括：

```text
test_backend_registry.py
test_backend_metal_adapter.py
test_backend_phase0.py
test_launch.py
test_tcenv.py
test_baselines.py
test_bench_limit.py
test_pressure.py
test_skills.py
test_symptom.py
test_xctrace.py
test_orchestrator_gpu.py
```

通用搜索、审查、observe、预算和任务接口测试不要放这里。Metal golden 数据在
`tests/support/golden/metal/`；新增 Metal 专属 fixture 也放那里。
