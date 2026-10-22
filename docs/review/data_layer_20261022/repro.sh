#!/usr/bin/env bash
# Read-only reproduction for docs/review/data_layer_20261022/README.md
#
# Safety:
#   - never runs `taic run` / `taic resume`
#   - never launches a GPU kernel
#   - never calls an LLM/API
#   - the only writes are under mktemp -d (removed on exit) and PYTHONDONTWRITEBYTECODE=1
#
# Usage:
#   ./repro.sh            # fast, ~10-20 s
#   SLOW=1 ./repro.sh     # also times run_detail/feature_grid (can take ~10 s on one run)
#
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" || exit 1

export PYTHONDONTWRITEBYTECODE=1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PY="${PYTHON:-python3}"
UV="${UV:-uv}"
SLOW="${SLOW:-0}"

hr() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

hr "0. repo state"
git rev-parse --short HEAD 2>/dev/null || true
git status --porcelain 2>/dev/null | head -20
printf 'python: '; command -v "$PY" || true
printf 'uv:     '; command -v "$UV" || true

hr "1. sizes"
find runs -type f 2>/dev/null | wc -l
du -sh runs data 2>/dev/null || true

hr "2. artifact breakdown (bytes)"
"$PY" - <<'PY'
import os
from pathlib import Path
root = Path("runs")
groups = {"transcript.jsonl": 0, "export(xml)": 0, "agent_work": 0, "candidates": 0, "other": 0}
counts = dict.fromkeys(groups, 0)
for p in root.rglob("*"):
    if not p.is_file():
        continue
    b = p.stat().st_size
    parts = p.parts
    if p.name == "transcript.jsonl":
        k = "transcript.jsonl"
    elif p.suffix == ".xml":
        k = "export(xml)"
    elif "agent_work" in parts:
        k = "agent_work"
    elif "candidates" in parts:
        k = "candidates"
    else:
        k = "other"
    groups[k] += b; counts[k] += 1
total = sum(groups.values()) or 1
for k in ("transcript.jsonl", "export(xml)", "agent_work", "candidates", "other"):
    print(f"{k:16s} {groups[k]/1e6:9.1f} MB  {counts[k]:6d} files  {100*groups[k]/total:5.1f}%")
print(f"{'TOTAL':16s} {total/1e6:9.1f} MB")
PY

hr "3. run dirs / protocol / agent sessions / event kinds"
"$PY" - <<'PY'
import collections, json
from pathlib import Path
runs = [p for p in Path("runs").rglob("*") if (p/"state.json").is_file() or (p/"result.json").is_file()]
proto = collections.Counter(); kinds = collections.Counter(); sessions = collections.Counter()
for r in runs:
    cfg = {}
    try: cfg = json.loads((r/"run_config.json").read_text())
    except Exception: pass
    proto[cfg.get("loop_protocol", "legacy/chat")] += 1
    t = r/"transcript.jsonl"
    if not t.is_file(): continue
    for line in t.open(errors="replace"):
        try: d = json.loads(line)
        except Exception: continue
        k = d.get("kind") or d.get("event") or "?"
        kinds[k] += 1
        if k == "agent": sessions[str(r)] += 1
print("run dirs:", len(runs))
print("protocols:", dict(proto))
print("event kinds:", dict(kinds))
print("agent sessions:", sum(sessions.values()), "in", len(sessions), "runs")
for n, r in sorted(((n, r) for r, n in sessions.items()), reverse=True):
    print(f"  {n:3d}  {r}")
PY

hr "4. transcript composition (largest legacy + a hmz run)"
"$PY" - <<'PY'
import collections, json
from pathlib import Path
cands = []
for t in Path("runs").rglob("transcript.jsonl"):
    cands.append((t.stat().st_size, t))
for size, t in sorted(cands, reverse=True)[:2]:
    kinds = collections.Counter(); sizes = collections.Counter()
    for line in t.open(errors="replace"):
        try: d = json.loads(line)
        except Exception: kinds["unparsed"] += 1; continue
        k = d.get("kind") or d.get("event") or "?"
        kinds[k] += 1; sizes[k] += len(line)
    print(f"{t}  {size/1e6:.2f} MB")
    for k, c in sizes.most_common(4):
        print(f"  {k:12s} {kinds[k]:6d} lines {c/1e6:7.2f} MB avg {c/max(1,kinds[k]):8.0f} B")
PY

hr "5. duplicate kernel/best sources"
"$PY" - <<'PY'
import collections, hashlib
from pathlib import Path
files = list(Path("runs").rglob("kernel.metal")) + list(Path("runs").rglob("best.metal"))
by = collections.defaultdict(list); total = 0
for p in files:
    try:
        b = p.read_bytes(); total += len(b)
        by[hashlib.sha256(b).hexdigest()].append(p)
    except OSError:
        pass
dups = {h: ps for h, ps in by.items() if len(ps) > 1}
extra = sum(len(ps[0].read_bytes()) * (len(ps)-1) for ps in dups.values())
print(f"files {len(files)}, bytes {total/1e6:.1f} MB, distinct {len(by)}, duplicated texts {len(dups)}")
print(f"duplicate copies beyond first: {extra/1e6:.1f} MB")
for h, ps in sorted(dups.items(), key=lambda kv: -len(kv[1]))[:3]:
    print(f"  {len(ps)} copies, {len(ps[0].read_bytes())} B: {ps[0]}")
PY

hr "6. index tables (read-only)"
"$PY" - <<'PY'
import sqlite3
con = sqlite3.connect("file:data/observe.sqlite?mode=ro", uri=True)
q = lambda s: con.execute(s).fetchall()
print("runs:", q("select count(*) from runs")[0][0])
print("turns:", q("select count(*) from turns")[0][0])
print("feedback_items:", q("select count(*) from feedback_items")[0][0])
print("claims:", q("select count(*) from feedback_claims")[0][0])
print("experiments:", q("select count(*) from experiments")[0][0], "arms:", q("select count(*) from arms")[0][0])
print("turn_bytes>0:", q("select count(*) from runs where turn_bytes>0")[0][0])
print("publication:", q("select publication,count(*) from runs group by publication"))
print("model:", q("select model,count(*) from runs group by model order by 2 desc limit 6"))
print("claims topic non-empty:", q("select count(*) from feedback_claims where coalesce(topic,'')<>''")[0][0])
print("claims premise non-empty:", q("select count(*) from feedback_claims where coalesce(premise,'')<>''")[0][0])
print("runs with experiment_id:", q("select count(*) from runs where coalesce(experiment_id,'')<>''")[0][0])
PY

hr "7. brief bytes + wall time"
if have "$UV"; then
  RUN="runs/apple-m4-max/conv2d-committees-20261020"
  [ -d "$RUN" ] || RUN="$(find runs -maxdepth 2 -type d -name '*conv2d*' | head -1)"
  if [ -d "$RUN" ]; then
    /usr/bin/time -p "$UV" run python -m taic.cli brief "$RUN" > "$TMP/brief.json" 2> "$TMP/brief.time" || true
    echo "run: $RUN"
    wc -c < "$TMP/brief.json"
    cat "$TMP/brief.time" | grep -E 'real|user' || true
    "$PY" - <<PY
import json
d=json.load(open("$TMP/brief.json"))
print("keys:", sorted(d))
print("gpu events:", len(d.get("gpu") or []), "bytes:", len(json.dumps(d.get("gpu") or [])))
print("candidates:", len(d.get("cands") or []), "paths:", list((d.get("paths") or {}).keys()))
PY
  else
    echo "SKIP: no run dir"
  fi
else
  echo "SKIP: uv not found"
fi

hr "8. run_detail bytes (temp index copy; no repo writes)"
if have "$UV"; then
  mkdir -p "$TMP/observe"
  cp data/observe.sqlite* "$TMP/" 2>/dev/null || true
  OBSERVE_DB="$TMP/observe.sqlite" AI_LOWERLING_OBSERVE_DIR="$TMP/observe" \
  "$UV" run python - "$SLOW" <<'PY' || true
import json, sys, time
from observe import query
rid = "apple-m4-max/conv2d-committees-20261020"
t0 = time.perf_counter()
d = query.run_detail(rid)
dt = time.perf_counter() - t0
if d is None:
    print("SKIP: run not in index"); sys.exit(0)
print(f"run_detail {rid}: {len(json.dumps(d, default=str))/1024:.1f} KB in {dt*1000:.0f} ms")
for k, v in sorted(d.items()):
    try: n = len(json.dumps(v, default=str))
    except Exception: continue
    if n > 500: print(f"  {k:18s} {n/1024:7.1f} KB")
if len(sys.argv) > 1 and sys.argv[1] == "1":
    t0 = time.perf_counter(); d2 = query.run_detail(rid); print(f"run_detail again: {(time.perf_counter()-t0)*1000:.0f} ms")
PY
else
  echo "SKIP: uv not found"
fi

hr "9. TASK.md sizes (per session copy)"
"$PY" - <<'PY'
import statistics
from pathlib import Path
sizes = [p.stat().st_size for p in Path("runs").rglob("*/agent_work/*/TASK.md")]
if not sizes:
    print("none"); raise SystemExit
print("count:", len(sizes), "max:", max(sizes), "median:", int(statistics.median(sizes)))
largest = max(Path("runs").rglob("*/agent_work/*/TASK.md"), key=lambda p: p.stat().st_size)
print("largest:", largest, largest.stat().st_size)
PY

hr "10. files per run (modern/hmz vs legacy)"
"$PY" - <<'PY'
import statistics
from pathlib import Path
rows = []
for r in Path("runs").rglob("*"):
    if not ((r/"state.json").is_file() or (r/"result.json").is_file()): continue
    n = sum(1 for _ in r.rglob("*") if _.is_file())
    has_agent = (r/"agent_work").is_dir()
    rows.append((n, has_agent, str(r)))
print("all runs median files:", int(statistics.median([n for n,_,_ in rows])), "max:", max(n for n,_,_ in rows))
hmz = [n for n,a,_ in rows if a]
print("runs with agent_work:", len(hmz), "median files:", int(statistics.median(hmz)) if hmz else "-")
for n,a,p in sorted(rows, reverse=True)[:5]: print(f"  {n:5d} agent_work={a}  {p}")
PY

hr "11. ledger token fields (hmz charge writes zero tokens?)"
"$PY" - <<'PY'
import json
from pathlib import Path
shown = 0
for t in sorted(Path("runs").rglob("transcript.jsonl"), key=lambda p: -p.stat().st_size):
    run = t.parent
    if not any(True for line in t.open(errors="replace") if '"agent"' in line):
        continue
    res = {}
    try: res = json.loads((run/"result.json").read_text())
    except Exception: pass
    print(run, "spent:", res.get("spent_usd"), "budget:", res.get("budget_usd"))
    print("  ledger summary:", (res.get("ledger") or {}))
    shown += 1
    if shown >= 3: break
PY

hr "done"
echo "No repo files were modified by this script (only $TMP, removed on exit)."
