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
    ├── npu_bundle_v4.json          # P1 bundle contract 向量（Go 独立重建）
    ├── npu_build_v5.json           # P1c build contract 向量（Go 独立重建 request/producer/evidence）
    └── evaluator_v1_baseline.desc  # v1 schema compatibility baseline，不是运行时输入
```

## 规则

- 重录用 `../../README.md` 里的 `--record --reason` 命令。
- golden 改动要和生成它的代码改动放在同一提交。
- 新增目录按 backend 命名：`golden/<backend_id>/`。

## 阶段验收包

`shared/` 下的阶段验收包是闭合历史边界，不因系统变更而改写、合并或重录。
包内 JSON、NDJSON、golden、`verify.py` 和 `manifest.sha256` 等证据不符时，一律报告
「闭合包被改写」，不能用新纪元豁免。Markdown 文档更新是单独的人工流程：必须明确批准
「文档更新，不是证据篡改」，在 `shared/acceptance-epochs/` 新增记录，并把该包 manifest
中的对应行更新到新字节；文档更新不能夹带其他 manifest 行或证据变化。

### 活跃锚点与纪元

- 当前提示指纹、任务包闭包文件清单和闭包摘要只由活跃测试从
  `tests/support/live_anchors.json` 读取。闭合 verifier 不读这份文件，也不读取
  「latest」别名；它们仍使用自己包内的历史 identity 和纪元记录。
- 提示面或任务包闭包变化时，更新这一份活跃锚点，并在
  `shared/acceptance-epochs/YYYYMMDD-<slug>.json` 追加一条纪元。纪元的
  `expected_drift` 必须绑定具体包、检查项、路径及前后值；旧记录不修改，旧值到新值的
  动态别名不允许。
- 从闭合 verifier 复制到 `tests/support/` 的库函数要在新模块 docstring 写明来源。
  原 verifier 不改；复制函数后续若有修复，在新纪元登记原 verifier 的漂移。

### 只读盘点与状态检查

```bash
uv run python -m tests.support.acceptance_inventory --json
uv run python -m tests.support.acceptance_status
```

这两条命令只读文件、AST、哈希和本地 Git 对象，不执行任何验收包的 `verify.py`。
状态检查把结果分为一致、已登记漂移、未登记和闭合包被改写；它不会把历史漂移登记
解释成旧 verifier 今天仍能通过。

### 退役与恢复

退役是经人逐个批准后从工作树移除包，Git 历史仍保留，不是删除历史证据。先运行：

```bash
uv run python -m tests.support.acceptance_status --retirement-check <package>
```

报告必须逐项处理活跃代码/测试引用、其他闭合包的字节钉与语义引用、文档链接、
`ARCHIVE_ROOTS` 和 AGENTS.md 当前状态唯一出处。AGENTS.md 的唯一出处判断由人完成；
History 规则允许把不再被活跃测试或活跃代码按路径打开的包移出工作区。
没有按名字保护，也没有数量上限。字节留在 git 历史里，登记在 `retired.json`。

批准后，在 `shared/retired.json` 登记包根、原因、接替者、批准人、最后完整提交、
对应的 tree、文件数/字节数、恢复命令和退休纪元；另写一条 `change_kind=retirement`
纪元登记其他闭合包因此产生的预期漂移。一次变更可以登记多个包，没有每季数量上限。
工具只检查和验证记录，不提供自动删除命令。

恢复使用记录中的命令：

```bash
git checkout <last_commit> -- <package>
```

`<tree>` 必须等于 `<last_commit>:<package>`，并且该 tree 仍存在于本地 Git 对象库；
浅克隆缺少对象时先取回历史，不能声称记录已可恢复。

### 工作区里还在的包

历史包的字节在 git 里，清单是 `shared/retired.json`。
工作区里留下的包，是因为测试还在读它们的输入，不是因为它们是最新阶段。
新阶段的证据可以封存；之后的测试不得按路径打开已经封存的目录。
当前提示指纹和任务包闭包只从 `tests/support/live_anchors.json` 读取。
