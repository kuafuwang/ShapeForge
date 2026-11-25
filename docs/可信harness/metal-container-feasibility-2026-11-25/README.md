# 真实 v2 Metal + 生产加固容器联合验收：平台阻塞（2026-11-25）

## 结论

用户明确要求做这项联合验收。实际检查后结论是 **`blocked_unsupported_platform`**，不是 `passed`：

> 当前生产容器执行面是 Linux Docker worker，而 Apple Metal/PyObjC、MLX 与 Darwin native Host Program 工具链不能在这条 Linux 容器路径里提供真实 Apple GPU 执行。仓库也没有一个完整、可部署的生产 Metal worker 镜像。

因此本轮没有执行真实 Metal 候选，没有真实设备正确性/性能证据，也没有把 process-worker 或 CPU fake backend 的结果改写成容器生产通过。

## 实际执行的容器探针

启动了本机 Docker daemon，并用生产同款加固参数运行真实容器：

- `--init --read-only --network none --cap-drop ALL --no-new-privileges`；
- 数值 UID 65534、pid/memory/cpu 限制、`/tmp` 与 `/scratch` tmpfs；
- 镜像 `taic-remote-grpc-sim:local`，image ID `sha256:aa1e7ca5afd9...`（完整值见 JSON）。

容器返回：

```json
{
  "sys_platform": "linux",
  "platform": "Linux-6.12.76-linuxkit-aarch64-with-glibc2.39",
  "metal_module": false,
  "mlx_module": false,
  "metal_library": null,
  "metal_error_type": "ModuleNotFoundError",
  "metal_error": "No module named 'objc'"
}
```

该镜像是既有 simulation 镜像，不是生产 Metal worker 镜像；这个探针只证明真实加固 Linux 容器的当前运行面，不把“缺 objc”单独冒充完整的设备证明。根本阻塞由平台支持范围和 worker 镜像内容共同确定。

## 代码层面的阻塞

当前容器路径为：

```text
job_worker_bridge
  → tcenv.remote.worker
  → ProductionBackend / V2LocalEvaluator
  → MetalSourceBuilder / Metal runtime
  → objc + Metal + MTLCreateSystemDefaultDevice
```

要真正运行当前 v2 Metal worker，镜像至少还需要当前 `taic`、`targets`、`backends` 闭包，以及 macOS 的 PyObjC/Metal framework。native Host Program 还要求 Darwin、Clang、`nm` 和 `size`。现有 Ubuntu Dockerfile 和 simulation 镜像不满足这些条件；Linux 上安装 Python 包也不能提供 Apple Metal framework。

官方边界：

- Docker 的 GPU 文档描述 NVIDIA GPU，不是 Apple Metal：<https://docs.docker.com/engine/containers/gpu/>。
- Apple Containerization 运行的是 Linux VM/container：<https://developer.apple.com/videos/play/wwdc2025/346/>。
- Apple `container` 的 GPU passthrough 仍是开放请求，没有可采用的 Metal 路径：<https://github.com/apple/container/issues/1511>。

## 后续如果要做设备隔离

不能把当前任务静默换成另一种技术并冒称容器验收。可选的新阶段是：

1. 为 Apple Metal 设计并验收原生 macOS Seatbelt/独立账号/VM 隔离，明确它不是 Docker 容器；
2. 对真实 Linux 设备 backend 单独实现驱动/设备节点与生产镜像，再验收该 backend 的容器路径；
3. 等 Apple/Docker 提供受支持的 Metal compute 通路后，再构建完整 Metal worker 镜像并重开本次联合验收。

## 清理与证据

Docker daemon 已停止，探针容器使用 `--rm` 已删除，没有新增镜像或遗留 worker。没有付费模型、搜索、NPU 或 speedup 发布。

逐字段结果见 [evidence.json](evidence.json)，原始容器输出见 [container-probe.json](container-probe.json)。
