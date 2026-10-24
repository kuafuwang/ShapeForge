# Shared tests

## 这篇讲什么

- 这里放 backend（后端）无关的测试：搜索、审查、observe、预算、信任、任务接口和
  通用策略。
- 换一个 backend，这条断言依然成立，就属于这里；需要 Metal 行为的测试放
  `tests/backends/metal/`。
- 新增共享 helper 放 `tests/support/`，不要放在本目录。

## 边界

不要 import `tcenv.backends.<id>.*`，也不要放 `.metal` 等 backend 专属 fixture。
遇到需要 Metal 行为的断言，移到 `tests/backends/metal/`。

| 内容 | 放哪 |
| --- | --- |
| 搜索、审查、observe、预算、信任、任务接口 | 本目录 |
| 通用策略、换 backend 后依然成立的断言 | 本目录 |
| `tcenv.backends.<id>.*` 的 import | 不放 |
| `.metal` 等 backend 专属 fixture | 不放 |
| 需要 Metal 行为 | `tests/backends/metal/` |

放置规则、导入、并行和失败日志见 `tests/README.md`；这条分层规则的可执行版本是
`tests/shared/test_structure.py`。
