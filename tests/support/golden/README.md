# Golden data

## 这篇讲什么

- golden（金样数据）按 backend（后端）分组：prompt、fixture 和期望产物都是某个
  backend 专属的，不能混在一起。
- 未来的 backend 在 `metal/` 旁边加一个 `golden/<id>/`，不要再把 backend 专属文件
  直接放到 `golden/` 根下。
- 读写的实现是 `tests.support.hw_golden`；录制和放置规则在 `../../README.md`。

```text
golden/
└── metal/
    ├── backend_phase0_kernel.metal
    ├── hw_off.json
    └── fixtures/
```

## 规则

- 重录用 `../../README.md` 里的 `--record --reason` 命令。
- golden 改动要和生成它的代码改动放在同一提交。
- 新增目录按 backend 命名：`golden/<backend_id>/`。
