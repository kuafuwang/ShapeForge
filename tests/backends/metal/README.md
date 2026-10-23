# Metal tests

这里放只对 Metal backend 有意义的测试：源码语法、编译、ABI、启动规格、运行时、
求值环境、Metal baseline、xctrace/profile 和 Metal golden。

不在这里放通用搜索/审查/observe/预算/任务接口测试（那些在 `tests/shared/`），
也不要在 `tests/shared/` 里 import `tcenv.backends.metal.*`。

当前目录就是清单，不要在这里维护第二份文件列表（迁移后曾因清单过期而误导）。
放置规则、导入约定、golden、并行和失败日志见 `tests/README.md`。
Metal golden 数据在 `tests/support/golden/metal/`。

判断标准：如果换一个 backend 这条断言仍然成立，它属于 `tests/shared/`；
需要 Metal 的编译、ABI、运行时或专属性能工具才放这里。
