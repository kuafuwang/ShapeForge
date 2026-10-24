#!/bin/sh
# Build the opt-in Linux patch-test image. TORCH_WHEEL is optional; when set,
# that wheel is copied into an ignored build-context directory for offline use.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
CONTEXT="$ROOT/tests/support/docker"
if [ -n "${TORCH_WHEEL:-}" ]; then
    if [ ! -f "$TORCH_WHEEL" ]; then
        echo "TORCH_WHEEL does not exist: $TORCH_WHEEL" >&2
        exit 2
    fi
    mkdir -p "$CONTEXT/wheels"
    rm -f "$CONTEXT"/wheels/torch-*.whl
    cp "$TORCH_WHEEL" "$CONTEXT/wheels/$(basename "$TORCH_WHEEL")"
fi
exec docker build --file "$CONTEXT/Dockerfile.patch-test" --tag "${IMAGE:-taic-patch-test:local}" "$CONTEXT"
