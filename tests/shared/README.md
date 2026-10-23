# Shared tests

这里放 backend 无关的测试：搜索、审查、observe、预算、信任、任务接口和通用
策略。不要 import `tcenv.backends.<id>.*`，也不要放 `.metal` 等 backend
专属 fixture。需要 Metal 行为的测试放 `tests/backends/metal/`。

放置规则、导入、并行和失败日志见 `tests/README.md`；这条分层规则的可执行版本是
`tests/shared/test_structure.py`。新增共享 helper 放 `tests/support/`。
