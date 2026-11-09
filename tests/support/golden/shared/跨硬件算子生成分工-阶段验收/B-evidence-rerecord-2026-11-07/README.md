# B1–B4 GPU 证据重录漂移处置（2026-11-07）

## 结论

本目录是一个**独立观察与处置 epoch**，不是 B1–B4 原验收包的替代品。

- 9 份发生 `median_us` 漂移的原始 JSON 已保存在 `raw/` 下，路径与 B 阶段一一对应；
- `B1–B4/evidence/gpu-*.json` 已恢复为 HEAD 原字节；
- 原漂移记录 `MT1b/b-evidence-drift.json` 未修改；
- 后续 B GPU 测试写出已改为独立 run 目录，不再自动覆盖历史 evidence。

这里同时保留“新观察”和“历史原证据”。没有把新计时用于 MT1b 或任何性能发布，也没有回滚删除观察。

## 文件

- `resolution.json`：逐文件的 HEAD 摘要、观察摘要、原始/新 median_us 与恢复摘要；
- `raw/B*/evidence/gpu-*.json`：发生漂移时的真实 JSON 原字节；
- `verify.py`：重算 raw/恢复文件、逐字段一致性、写出边界和 manifest；
- `manifest.sha256`：本目录全部证据文件摘要。

## 后续写出规则

`tests/support/migration_gpu_evidence.py::write_gpu_observation` 默认每次写入独立临时目录。若需要形成新的性能 epoch，设置 `SHAPEFORGE_B_EVIDENCE_RUN_DIR` 为**封闭 evidence 之外的新目录**，然后独立复核和归档。将该变量指向旧 `B*/evidence/` 会被拒绝。

任何新的性能 epoch 都需要单独的 GPU 授权、安静窗口与 fresh evidence，不把本目录中的观察直接当作已发布结果。

## 复核

```bash
uv run python 'docs/架构/跨硬件算子生成分工-阶段验收/B-evidence-rerecord-2026-11-07/verify.py' --check
```

本验证只读文件并运行静态测试，不执行 GPU、模型或历史验收重写。
