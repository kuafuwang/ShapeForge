# Backend tests

每个 backend 一个目录：

```text
tests/backends/<backend_id>/
```

这里放只对该 backend 有意义的测试：源码语法、编译、ABI、运行时、启动规格、
专属性能工具和 golden fixture。不要在这里测试通用搜索/审查/observe 逻辑；
那些放 `tests/shared/`。不要把 Metal 测试放到其他 backend 的目录。

放置规则、导入、并行和失败日志的完整版见 `tests/README.md`；分层不变量由
`tests/shared/test_structure.py` 执行。
