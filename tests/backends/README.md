# Backend tests

## 这篇讲什么

- 每个 backend（后端）一个目录，路径是 `tests/backends/<backend_id>/`。
- 这里只放只有这个 backend（后端）才能验证的测试：源码语法、编译、ABI、运行时、
  启动规格、专属性能工具和 golden fixture。
- 通用搜索、审查、observe 逻辑不放这里；它们属于 `tests/shared/`。

```text
tests/backends/<backend_id>/
```

具体边界：

| 放这里 | 不放这里 |
| --- | --- |
| 源码语法、编译、ABI、启动规格、运行时 | 通用搜索、审查、observe 逻辑（放 `tests/shared/`） |
| 求值环境、专属性能工具、golden fixture | 其他 backend 的测试 |

不要把 Metal 测试放到其他 backend 的目录。

放置规则、导入、并行和失败日志的完整版见 `tests/README.md`；分层不变量由
`tests/shared/test_structure.py` 执行。
