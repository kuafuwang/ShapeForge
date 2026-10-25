# Production evaluator frontend image (no simulated backend, no test files).
#
# Build from the repository root:
#   docker build -f deploy/remote-evaluator/Dockerfile -t tcenv-evaluator:<version> .
#
# The image contains the gRPC frontend and the production worker entry
# (tcenv.remote.worker).  Device-specific backends add their own runtime
# dependencies in a derived image; this base installs only CPU Python and the
# optional gRPC/protobuf group.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/taic

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages \
        "grpcio==1.69.0" "grpcio-status==1.69.0" "protobuf==5.29.3" "numpy>=2.0"

# The task registry needs a numerical backend; CPU PyTorch is enough for the
# production fingerprint/provider and non-device test hooks.  Device-specific
# images replace or extend this layer.
RUN python3 -m pip install --break-system-packages \
        --index-url https://download.pytorch.org/whl/cpu torch

COPY tcenv /opt/taic/tcenv
COPY tasks /opt/taic/tasks
COPY pyproject.toml /opt/taic/pyproject.toml
COPY deploy/remote-evaluator/entrypoint.sh /usr/local/bin/tcenv-evaluator
RUN chmod 0755 /usr/local/bin/tcenv-evaluator

# Unprivileged runtime; mount the state volume writable and config read-only.
USER 10001:10001
WORKDIR /opt/taic
EXPOSE 7443
ENTRYPOINT ["/usr/local/bin/tcenv-evaluator"]
