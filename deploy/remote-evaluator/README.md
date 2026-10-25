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
  mount the Docker socket, device nodes or host HOME.

Example:

```bash
docker build -f deploy/remote-evaluator/Dockerfile -t tcenv-evaluator:2026-10-25 .
docker run --rm -p 7443:7443 --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --user 10001:10001 \
  --mount type=bind,src=/srv/tcenv/config,dst=/etc/tcenv,readonly \
  --mount type=bind,src=/srv/tcenv/state,dst=/var/lib/tcenv/state \
  tcenv-evaluator:2026-10-25 \
  --listen 0.0.0.0:7443 \
  --state-dir /var/lib/tcenv/state \
  --site-id npu-node-01 --device-id npu-node-01-dev0 \
  --backend npu --backend-version 1 --target-id npu-node-01 \
  --worker-build build-2026-10-25 --calibration-digest <64 hex> \
  --task-pack-root /etc/tcenv/task-pack \
  --grants-file /etc/tcenv/grants.json \
  --signing-keyring-file /etc/tcenv/keyring.json \
  --certificate-file /etc/tcenv/tls/server.pem \
  --private-key-file /etc/tcenv/tls/server.key \
  --client-ca-file /etc/tcenv/tls/client-ca.pem
```

`--check` validates the mounted files, task pack digest/fingerprints, grants
and active signing key without opening a listener or the state database.
