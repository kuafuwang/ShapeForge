# P2-K：conv2d small+paper portfolio fresh

- epoch：`p2-k-conv2d-portfolio-fresh-1`
- parent HEAD：`1365473c2f3d26ed882946a2fffce133d2128df0`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`

本阶段确认 P2-J 的完整 small+paper 候选树，不改 P2-I/P2-J 的任何历史文件。freeze 为 `db264da7…`，validation 为 `8ceace0e…`。四个 fresh subject 全部 `confirmed`、identity `match`：

- small benchmark：3.4935 µs，Torch-MPS 5.7140×、MLX 0.9111×；
- paper benchmark：611.2739 µs，Torch-MPS 1.2739×、MLX 1.5940×；
- small golden 与 paper golden：六个 case 和独立 fresh identity 均通过。

两档 benchmark 都满足 1.10 延迟门。small 仍未稳定超过最佳基线，不能写成最佳基线加速或生产发布。

`evidence/validated-manifest.json` 是由四个 subject 和独立 activation expectations 重建的完整覆盖 manifest，digest `f31ebdc787783b0ff7d1cff138b22f52992b2ff53666acf1ac3d755169fd88a6`。它保持 `status=validated`、`activation_generation=0`，没有 active pointer 或发布行为。真实 Metal 重放再次执行四个 fresh subject，并逐字段重建同一 manifest。

`evidence/packed-half-probe.json` 保留三次交替 A/B 的负结果：新版中位数 3.3685 µs、相对 MLX 0.9483×，未稳定超过 1.0×，`adopt_candidate=false`。P2-J small candidate 不被这次实验替换。

`verify.py` 校验 79 文件闭包、P2-I 71 文件与 P2-J 22 文件的历史字节、fresh suite、validated manifest 和负结果，然后运行 2 个 CPU 节点和 1 个真实 Metal 重放节点。预期最终行：

`P2-K acceptance (2 CPU + 1 real Metal replay; validated only, no activation/publication)`。

本阶段尚未声称独立 reviewer 签署。
