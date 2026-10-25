#!/bin/sh
# Production gRPC evaluator entry point.  All deployment files (TLS, grants,
# keyring, task pack) are mounted read-only; the state directory is persistent.
set -eu
exec python3 -m tcenv.remote.grpc_service "$@"
