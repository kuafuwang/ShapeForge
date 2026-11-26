# v2 remote 扩展交付形态：真实 VM→Mac 验收（2026-11-25）

## 范围与结论

本轮完成了此前未验证的三种 v2 remote 交付形态：

| 形态 | 任务 / Track | candidate SHA-256 | 真实结果 |
| --- | --- | --- | --- |
| MSL bundle | `affine_ramp/fp16/small`，`msl-bundle/1` | `8d265982fe191b9fc6b498571df9b2869d6f89cda83198edfbe1e1be5f1f249d` | `ok`，1 case，correctness/safety 通过 |
| native Host Program | `varlen_kda/fp16/mixed`，`msl-bundle/1` + `host/tiling.c` | `5e018ca1d22285c176366a5283313b1a1b64d6dba2bffd4835326c3fc777b521` | `ok`，9 个 L1 case；native C99 planner、4 stages |
| TileLang | `relu/fp16/small`，`tilelang-pipeline/3` | `c702520f48c0cc5a99ede4e13e0f442c52aa39b059abaae0d48ea72f7494c872` | `ok`，6 case（含 IEEE）；TileLang 0.1.15、builder version 3 |

每个请求都在真实 Ubuntu VM 客户端经 mTLS 提交到 Mac Metal worker，再对同一请求执行 replay。每个请求的 fresh result 与它自己的 replay 逐项一致（`replay_identity_match=true`）；三种形态各有独立的 attempt、receipt 和 lease fence（见下文「协议与身份」的清单），evaluator epoch 相同。replay 没有创建新 attempt。

**边界：这是 quick/formal correctness 与 durable replay 验收，不是性能确认或 speedup 发布。** `cache_policy=disabled`、`measurement_kind=measured`、`publication_allowed=false`。版本化 remote envelope 的 fresh confirmation/activation 尚未实现，会在 freeze/expected/build 之前被 `UnsupportedContractCapability` 显式拒绝，不能降级成未版本化 candidate SHA。

## 协议与身份

六 RPC、Protobuf 字段号、HMAC、BlobChannel 和 task-pack schema 未改。普通 MSL source 与历史未版本化 envelope 的本地编码、manifest 和候选身份保持原样。

extended envelope 新增可选 `taic-track-version`：

- 版本化 manifest 使用 `tcenv.candidate.artifact-manifest.2`，绑定 `track_id/track_version`；
- remote extended 在 `ExternalEvaluationRequest` 准入时强制要求版本，未带版本的 envelope 返回 `INVALID_ARGUMENT`，没有 durable job/attempt；
- worker、projection、lifecycle 和 build 全程保留版本；最终满足：

```text
request.candidate_sha256
== report.build.candidate_sha256
== report.build.provenance.source_sha256
```

- TileLang `/3` 只在内部 `/2` DSL source-contract 校验时去掉外层版本；实际 source-only 构建、native descriptor 和独立 artifact audit 仍使用外层版本化 candidate 身份。

最终真实 staging 身份：

- task-pack digest：`051d0d65a03797ae7305a48382fdc2e16a48d19f57d40ae884fd6bcab1da7b1e`
- worker build：`v2-extended-envelope-staging-r2`
- evaluator epoch：`ee5392d534b391139420b129ce206b30eb1221269daa4d7df5daf09947aab690`
- attempts：`attempt-1-b1b1b7b0ff82446c`、`attempt-2-6fc4133130bb4ccf`、`attempt-3-c6ff8b4868b14d3a`
- receipts：`ca3cd6a0d021121e223ac14f7adc8901baefe5952cf00c7791dd20b4b5d01e71`、`3fa20fc30dc854d0ee5aa6b29b198dbc860f0fc5cfbdc8c397b1825c6ac01bae`、`a8c1b820eeec85168b41105b4136fd9be3a85957c603f94ddaa89c21979ee4fb`

## 本机与远程交叉核对

同一版本化候选字节先在本机真实 Metal 通过，再发送到远端。三条路径的本机/远程 `native_digest` 和 `artifact_set_digest` 全部一致：

| 形态 | native_digest | artifact_set_digest |
| --- | --- | --- |
| MSL bundle | `4eb4bbd390bdfd3b55068cfdb734003ab0c33da44295398b1255e3fc2a8143a8` | `09209282514119a0375bb495fb4b539559ffdd5a90a1a42be0fcb68399742440` |
| Host Program | `b17c31e7fa8f669012540d9d1c89f514b42424922cb552870e97d995edcedc12` | `aaaa62853188d1e2cc475b1bc18419e55e0b3563e31bc302f0ca848f46d767ab` |
| TileLang | `a53090c72b2936ababe6fb9f8bbe52050be5fde32d92dc98be467f8e358be874` | `41265bd80279a4243cce3359de38fe97de170c2b8f8c007091623e100257dceb` |

逐字段原值见 [local-remote-identity.json](local-remote-identity.json)。

## 代码锚点与回归

- 功能提交：`553be0fe48b8b04a5fee5fd74f7e9ecb52819dc7`；binary 兼容修复：`236010efa5ec72baa461dacdd570f90fba75ff1b`。
- 当前闭包仍为 **78 文件**，摘要 `99acb3a63f4abceabd73ad80ef457f1a1eb7ed02e740dec95756b7497269a8c9`；prompt 未变。最新纪元：`20261125-remote-binary-source-compat.json`。
- 聚焦 remote/track/TileLang 回归 **119 passed**，扩展 CPU 回归 **211 passed**（两处当时的闭包旧锚点除外，随后已登记）。
- 完整安全套件一次为 **5258 passed、9 skipped、9 failed**。其中 binary source 与清微当前重放已在随后修复并复验；scorecard、observe SSE、observe endpoints、container 四个时序项串行复验通过，container 在同一 Python 下对父提交也通过。剩余分类为两个历史 trust 锚点和 anchor-only worktree HEAD，不能写成全套全绿。
- 最新治理、闭包与清微复验 **47 passed**；证据自校验确认三份候选、request/build/provenance、replay、native/artifact 对照一致。
- 最终独立复核 `done=true`（[review-final.json](review-final.json)），范围仅限 quick/formal；初次两个 blocker 及后续 binary 兼容问题均已关闭。

## 环境与清理

- Mac：MetalToolchain 27A266a；固定 TileLang 0.1.15 staging，package manifest `06f3b0adee7abce77af1111cb36e704b4b23c3cbe38d095dd62cce86509c11cc`。
- VM：Ubuntu 22.04.3 / aarch64，Python 3.10.12；隔离客户端使用 grpcio 1.84.1、grpcio-status 1.71.2、protobuf 5.29.6、tomli 2.2.1。没有修改 VM 系统 Python，也没有安装 Metal/Torch。
- 服务、CA/client key、token、grant、HMAC keyring、专用 state 和 VM 客户端目录均已删除；端口不再监听，没有遗留 worker，VM 已停止。见 [cleanup.json](cleanup.json)。

## 证据文件

- [remote-summary.json](remote-summary.json)：三个请求/replay 的摘要与身份。
- `remote-results/`：完整远程 report、receipt、evidence、provenance。
- `candidates/`：实际发送的三份版本化 envelope 和本机摘要。
- [local-remote-identity.json](local-remote-identity.json)：本机/远程 native 与 artifact digest 对照。
- [environment.json](environment.json)：客户端、设备、worker 和测量边界。
- [task-fingerprints.json](task-fingerprints.json) 与 `task-pack/`：实际登记的合同指纹。
- [client_submit.py](client_submit.py)：不含凭据的请求/replay 客户端脚本。

没有付费模型、搜索、真实 NPU 或第二颗 Metal 设备。容器生产隔离、reuse/refresh cache、真实 NPU 和性能发布仍不属于本轮完成范围。
