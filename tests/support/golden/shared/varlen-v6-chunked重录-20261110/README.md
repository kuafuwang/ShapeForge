# V6 chunked 六档 sidecar 重录（2026-11-10）

这是一次新的真实 Mac 重录，不是对 `varlen-v6-chunked验收/` 的改写。

旧包的 `--check` 停在 `confirmation seed is not derived from this validation plan`：
归档 freeze `67cd0926…` / plan `765fb710…` 与当前代码重建的 `660d06d4…` / `e7278968…`
不是同一次派生。旧 sidecar 继续保留为历史测量。本目录只接收这次按当前
`deployment_pack_digest` 重新确认的六档结果。

`publication_allowed` 不因这次重录打开。NPU-P0 仍 blocked。
