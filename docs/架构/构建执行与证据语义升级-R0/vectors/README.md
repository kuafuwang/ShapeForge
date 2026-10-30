# R0 canonical identity vectors

用途：把 plan §3.1 的身份**形状**和摘要算法钉成可复算向量，供 R1–R5 对照。它们**不是**实现，
也不定义任何语义规则：

- 数值（`flops`、`min_bytes`、容差、`EPS`）是夹具值，不定义 R1 的记账规则或容差来源。
- `lowering_profile.json` 是 R0 提出的字段闭集**示例**；R1 登记最终字段集时以 schema 为准，
  字段集或取值规则变化必须同时升 `lowering_profile_digest`（等价于“升向量”）。
- `task.json` 里嵌的 `triton_source`/`reference_source` 是固定占位文本，不是真实模板内容。

## 规则（与 plan §3.1 一致）

1. `digest = SHA256(RFC8785(canonical_document))`，64 位小写十六进制。
2. 对象键必须是非空 ASCII 字符串；字符串按原码点哈希，**不做 Unicode 归一化**；重复键由解析器拒绝。
3. 整数值直接用 JSON 整数（IEEE-754 安全范围内）。非整数数值先用最短往返 `repr` 变成**十进制字符串**：
   `1e-06`、`0.002`、`-0.0`；要求 `float(s) == value` 且 `repr(float(s)) == s`。NaN/±Inf 拒绝。
4. `0.0` 与 `-0.0` 是不同 canonical 值；`1`（整数）与 `"1.0"`（浮点字符串）是不同 canonical 值。
5. 同一逻辑字段的类型由 schema/规则版本固定，不能让同一字段在两次实例化中时而 int 时而 float。

`encoding_examples.json` 是第 3、4 条的可执行样例；`verify.py` 会逐条复算。

## 文件与摘要

| 文件 | 身份 | 说明 |
| --- | --- | --- |
| `documents/opspec.json` | `opspec_digest` | 完整 OpSpec + `semantic_rules_version` + `work_accounting_version` |
| `documents/lowering_profile.json` | `lowering_profile_digest` | 完全解析的 profile + `lowerer{id,version}` + `variant_factory`（非 variant 为 `null`） |
| `documents/semantic_instance.json` | （嵌入 `task.json`） | `SemanticInstance.fingerprint_document()` 的闭集形状 |
| `documents/specialization.json` | （嵌入 `task.json`） | `Specialization.fingerprint_document()`：Contract 全字段，buffer `value=null` 不省略 |
| `documents/task.json` | `task_fingerprint`（v2） | 任务身份：完整派生投影 + `description` + 渲染后的源码 |
| `documents/baseline_implementation.json` | `baseline_implementation_digest` | 完整 `Task.baselines` 登记实现（不含可用性/库版本） |
| `documents/baseline_environment.json` | `baseline_environment_digest` | 测量进程声明的可用性与库版本 |
| `documents/task_pack.json` | `task_pack_digest` | `{schema_version, files:[{path,size,sha256}]}`，与 `tcenv/remote/task_pack.py` 同形 |
| `documents/evaluation.json` | `evaluation_fingerprint` | 组合 `task_fingerprint`、pack、gen/reference/error_scale、两份 baseline 摘要 |

`digests.json` 记录每个身份文档的路径与摘要；`../manifest.json::vectors` 内嵌同一份。

## 复算

```bash
# 重新生成（stdlib，不 import tcenv）：会重写 documents/*.json 与 digests.json
uv run python "docs/架构/构建执行与证据语义升级-R0/gen_vectors.py"

# 只校验（独立序列化器，逐一比对摘要与编码规则）
python3 "docs/架构/构建执行与证据语义升级-R0/verify.py"
```

两条路径必须给出相同摘要；R0 冻结时已用仓库的
`tcenv.remote.contracts.canonical_digest_document` 做过第二次独立复算并通过。

## 变更纪律

- 向量变化必须与“规则/字段集变化”同批出现，并在提交信息里写明原因；不得为了对齐实现而改向量。
- 将来要让 float 进入这些摘要（例如某字段必须保留 IEEE 语义），必须先给数字序列化补跨实现测试向量，
  再改规则；不得另写一个支持 float 的 RFC 8785 序列化器（plan §3.1 末段）。
