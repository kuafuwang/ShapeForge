# Remote evaluator container

This directory contains the production image for the external
`tcenv.remote.v1` gRPC frontend and its entry point.

- The image runs `python3 -m tcenv.remote.grpc_service`; it never copies
  `tests/` and the production entry refuses test worker modules and simulated
  measurement identities.
- TLS/mTLS is mandatory. Mount `certificate.pem`, `certificate.key`,
  `client-ca.pem`, `grants.json`, `keyring.json` and the read-only task pack;
  see `docs/架构/远程评测部署与轮换.md`.
- Mount the state directory as a writable persistent host volume with mode
  0700 and make it owned by the container uid (default 10001). SQLite WAL,
  `synchronous=FULL` and the single-writer owner lock require a local
  filesystem, not NFS/9p or an ephemeral container layer.
- `docker run --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m` with only the
  state volume writable is the intended shape; drop all capabilities, never
  mount device nodes or host HOME.
- **Worker execution (default: one disposable container per attempt).**
  `python -m tcenv.remote.grpc_service` now defaults to
  `--worker-execution container`. That requires the *frontend* process to reach
  the Docker Engine (local socket or `--docker-host`) and a `docker` CLI on its
  `PATH`. The base image above does **not** install the Docker CLI: run the
  frontend directly on the evaluator host (recommended; service account in the
  `docker` group, state dir 0700) or build a derived image that installs the
  Docker CLI and mount the socket into the *frontend* only. The Docker socket is
  root-equivalent: the frontend is the trusted lifecycle supervisor, and the
  supervisor never mounts it (or any bind mount) into a worker container.
  A second, separate worker image is required
  (`--worker-image`, pinned with `--worker-image-digest`) and must contain this
  repository's `tcenv` and the production worker entry; it runs unprivileged,
  read-only, with no network and no socket. `--worker-build` must end with
  `+img:<the same digest>` so the signed provenance binds the actual image.
  Process mode remains available for local development only:
  `--worker-execution process --allow-process-worker`.

Example:

Container execution (recommended shape: frontend on the evaluator host, local
Docker Engine; the worker image is separate):

```bash
# Frontend (host install, trusted, in the docker group):
python -m tcenv.remote.grpc_service \
  --listen 0.0.0.0:7443 \
  --state-dir /var/lib/tcenv/state \
  --worker-execution container \
  --worker-image registry.example/tcenv-worker:1.0 \
  --worker-image-digest <64 hex image id> \
  --worker-build 'evaluator-1.0+img:<the same 64 hex digest>' \
  --worker-python python3 \
  --site-id npu-node-01 --device-id npu-node-01-dev0 \
  --backend npu --backend-version 1 --target-id npu-node-01 \
  --calibration-digest <64 hex> \
  --task-pack-root /etc/tcenv/task-pack \
  --grants-file /etc/tcenv/grants.json \
  --signing-keyring-file /etc/tcenv/keyring.json \
  --certificate-file /etc/tcenv/tls/server.pem \
  --private-key-file /etc/tcenv/tls/server.key \
  --client-ca-file /etc/tcenv/tls/client-ca.pem
```

If the frontend itself must run in a container, add the Docker CLI to a derived
image and mount `/var/run/docker.sock` into the frontend (not the worker); the
state and config mounts stay as above.

Local development (process worker, not for production):

```bash
python -m tcenv.remote.grpc_service ... \
  --worker-execution process --allow-process-worker
```

`--check` validates the mounted files, task pack digest/fingerprints, grants
and active signing key without opening a listener or the state database.
