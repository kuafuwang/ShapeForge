# Golden data

## 这篇讲什么

- golden（金样数据）按 backend（后端）分组：prompt、fixture 和期望产物都是某个
  backend 专属的，不能混在一起。
- 未来的 backend 在 `metal/` 旁边加一个 `golden/<id>/`，不要再把 backend 专属文件
  直接放到 `golden/` 根下。
- backend 无关的对外 evaluator 合约向量放在 `remote/`；它不是新 backend，也不包含硬件测量。
- backend 读写的实现是 `tests.support.hw_golden`；对外协议只读 loader 是
  `tests.support.remote_contract`。录制和放置规则在 `../../README.md`。

```text
golden/
├── metal/
│   ├── backend_phase0_kernel.metal
│   ├── hw_off.json
│   └── fixtures/
└── remote/
    ├── evaluator_v1.json
    ├── npu_blob_v3.json            # P0b blob-reference contract 向量（Go 独立重建）
    └── evaluator_v1_baseline.desc  # v1 schema compatibility baseline，不是运行时输入
```

## 规则

- 重录用 `../../README.md` 里的 `--record --reason` 命令。
- golden 改动要和生成它的代码改动放在同一提交。
- 新增目录按 backend 命名：`golden/<backend_id>/`。
