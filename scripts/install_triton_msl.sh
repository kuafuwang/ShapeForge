#!/bin/zsh
# Isolated environment with Triton + the community Metal backend triton-msl (used only as an optional *compiler baseline*).
# Nothing here touches the main environment. Resumable: every download uses curl -C - with retries.
#
# Trust notes (also in the README):
#   * triton-msl 0.3.0: the PyPI wheel; its sha256 is pinned below and was checked against the official PyPI metadata.
#   * Triton 3.7.0 @ 4da2e268: the unofficial prebuilt wheel published by the triton-msl author (upstream has no macOS wheel).
#     Its sha256 is printed and stored in .venvs/triton-msl/TRITON_WHEEL.sha256 for later comparison.
#   * Python packages come from the Tsinghua PyPI mirror (override with MIRROR=...).
setopt nonomatch
set -e
cd "$(dirname "$0")/.."
MIRROR=${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
VENV=.venvs/triton-msl
PY=$PWD/$VENV/bin/python
DL=${DL:-/tmp/triton_dl}
mkdir -p $DL

TMSL_SHA=79d3a5545538636dc011d513bb35f232ffa757260354424db163aedeb685aee1
TMSL_WHL=triton_msl-0.3.0-cp314-cp314-macosx_15_0_arm64.whl
TRITON_WHL="triton-3.7.0+git4da2e268-cp314-cp314-macosx_15_0_arm64.whl"
TRITON_URL="https://github.com/bledden/triton-msl/releases/download/triton-wheel-3.7.0-cp314-macos-arm64/$TRITON_WHL"
TRITON_SIZE=142136193

[ -x $PY ] || { echo "[1/5] venv"; uv venv --python 3.14 $VENV; }
echo "[2/5] dependencies (mirror)"
uv pip install --python $PY --default-index $MIRROR numpy "torch>=2.12" mlx pyobjc-core pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders pyobjc-framework-Cocoa pytest

echo "[3/5] triton-msl 0.3.0 (hash pinned)"
if [ ! -f $DL/$TMSL_WHL ]; then
  URL=$(curl -s https://pypi.org/pypi/triton-msl/0.3.0/json | python3 -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename']=='$TMSL_WHL'][0])")
  until curl -fL -C - --retry 20 --retry-delay 5 --connect-timeout 20 -o $DL/$TMSL_WHL "$URL"; do sleep 5; done
fi
echo "$TMSL_SHA  $DL/$TMSL_WHL" | shasum -a 256 -c -
uv pip install --python $PY --no-deps $DL/$TMSL_WHL

echo "[4/5] Triton 3.7.0 prebuilt wheel (142 MB, resumable)"
until [ "$(stat -f %z "$DL/$TRITON_WHL" 2>/dev/null || echo 0)" = "$TRITON_SIZE" ]; do
  curl -fL -C - --retry 20 --retry-delay 5 --connect-timeout 20 -o "$DL/$TRITON_WHL" "$TRITON_URL" || sleep 5
done
SHA=$(shasum -a 256 "$DL/$TRITON_WHL" | cut -d' ' -f1)
echo "triton wheel sha256: $SHA"
echo "$SHA  $TRITON_WHL" > $VENV/TRITON_WHEEL.sha256
uv pip install --python $PY --no-deps "$DL/$TRITON_WHL"

echo "[5/5] smoke test (@triton.jit needs a real source file)"
cat > $DL/smoke.py <<'PY'
import torch, triton, triton.language as tl, triton_msl
print("triton", triton.__version__, "| torch", torch.__version__, "| mps", torch.backends.mps.is_available())

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

n = 4096
x = torch.randn(n, device="mps"); y = torch.randn(n, device="mps"); o = torch.empty_like(x)
add_kernel[(n // 256,)](x, y, o, n, BLOCK=256)
torch.mps.synchronize()
print("vector add max error:", (o - (x + y)).abs().max().item())
PY
$PY $DL/smoke.py
echo "DONE"
