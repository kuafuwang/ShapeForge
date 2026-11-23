# v1 正式退役：v1-track 批次

用户于 2026-11-23 明确要求正式退役 v1。本批次只处理可以独立闭合的 **v1-track 归档代码包和它的直接测试入口**，不把后续 specimen 迁移伪装成已经完成。

## 恢复点

- last commit：`bfd58fe716c90c533d948aee1fad68dbf9557263`
- `tests/support/golden/shared/v1-track` tree：`c1a1a61fdcb714ae3508a4769e3d933706b28846`
- 300 个 Git 跟踪文件，3,021,474 bytes（工作区额外的 pycache 不计入）
- 恢复命令：

  ```bash
  git checkout bfd58fe716c90c533d948aee1fad68dbf9557263 -- tests/support/golden/shared/v1-track
  ```

纯 v1 的 `tests/shared/test_track_binding.py` 同批移除，其历史 blob 为 `eb033f32db92313e6d24f75ac4d42f2431db4549`。恢复命令：

```bash
git checkout bfd58fe716c90c533d948aee1fad68dbf9557263 -- tests/shared/test_track_binding.py
```

`retired.json` 没有新增行：该文件的治理合同只接受验收包根，`v1-track` 是归档代码而不是 inventoried acceptance package root，单个测试文件也不是 tree。这里以本文作为合规的代码退役记录，不伪造清单条目。

## 活跃入口变化

- 删除纯 v1 的 `test_track_binding.py`；
- `test_opspec_v2.py`、`test_opspec_varlen.py`、`test_task_contract_v2.py` 移除只针对该 archive 的用例，其余 v2 测试保留；
- inventory/structure 不再为 `v1-track` 保留特殊兼容分支；
- 没有活跃测试或生产代码再按路径打开 `tests/support/golden/shared/v1-track`。

本批验证：**146 passed**，覆盖上述三个 v2 测试文件、structure 和完整 acceptance governance。`git diff --check` 通过。

## 尚未完成的退役工作

本批不等于 v1 已全部消失。`tests/support/v1_tasks.py`、`v1_specimen.py`、旧报告/只读重建以及生产中的 specimen/legacy 分支仍需后续明确迁移或删除；这些不能被本批的目录删除冒称完成。下一阶段应逐个替换共享测试的 v2 fixture，再移除剩余生产 fallback，而不是一次性删除共享机制测试。
