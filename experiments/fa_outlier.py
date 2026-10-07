"""Read-only: what the FlashAttention 'incorrect' failures actually are.

Almost every one in the October runs was a single element of 1,048,576. This script counts them from the candidate reports already on disk
and writes ``runs/fa_outlier/REPORT.md``. It does not change a tolerance and it does not run a kernel.

    uv run python -m experiments.fa_outlier
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTLIER = re.compile(r"case '(\w+)':\s*(\d+)/(\d+) elements out of tolerance \(worst scaled error ([0-9.]+)(?:, allowed ([0-9.]+))?\)")


def main() -> int:
    rows = []
    for report in REPO.glob("runs/**/candidates/*/report.json"):
        try:
            doc = json.loads(report.read_text())
        except (OSError, ValueError):
            continue
        if "flash_attention" not in str(doc.get("task", "")):
            continue
        if doc.get("status") not in ("incorrect", "unsafe"):
            continue
        message = doc.get("message") or ""
        m = OUTLIER.search(message)
        run = report.parents[3].name if len(report.parents) > 3 else report.parent.name
        rows.append({
            "run": run,
            "label": report.parent.name,
            "status": doc.get("status"),
            "case": m.group(1) if m else "",
            "bad": int(m.group(2)) if m else None,
            "n": int(m.group(3)) if m else None,
            "scaled": float(m.group(4)) if m else None,
            "message": " ".join(message.split())[:220],
        })
    by_case = Counter((r["case"] or "(no case parsed)") for r in rows)
    one = [r for r in rows if r["bad"] == 1]
    lines = [
        "# FlashAttention incorrect and unsafe failures",
        "",
        f"{len(rows)} candidate reports with status incorrect or unsafe. {len(one)} of them name exactly 1 element out of tolerance.",
        "",
        "| case | reports |",
        "| --- | ---: |",
    ]
    for case, n in by_case.most_common():
        lines.append(f"| {case} | {n} |")
    scaled = sorted(r["scaled"] for r in one if r["scaled"] is not None)
    if scaled:
        lines += ["", f"The 1-element failures have scaled error from {scaled[0]:.3f} to {scaled[-1]:.3f} (allowed is 1)."]
    lines += [
        "",
        "This is consistent with an online-softmax boundary (one element, just over the tolerance) and not with a kernel that is simply wrong.",
        "The tolerance is not loosened on this evidence: a scaled error of 1.03 and one of 1.48 are both over the line the reference sets,",
        "and the reports do not show whether the same index fails again on a second seed. That check is a fresh evaluation, not done here.",
        "",
    ]
    out = REPO / "runs" / "fa_outlier"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (out / "REPORT.md").write_text(text)
    (out / "data.json").write_text(json.dumps(rows, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
