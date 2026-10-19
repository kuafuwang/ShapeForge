"""Re-send truncated replies that contained no kernel, four ways, and compare which one produces a kernel.

The four arms, each a fresh request built from the original messages:

* ``as_sent``       the messages exactly as they were sent (the control);
* ``no_reasoning``  ``reasoning_content`` removed, and an empty truncated assistant turn dropped;
* ``low_effort``    ``no_reasoning``, with ``reasoning_effort`` set to ``low``;
* ``off_effort``    ``no_reasoning``, with ``reasoning_effort`` set to ``off``;
* ``write_now``     ``no_reasoning`` plus one line telling the model to write the kernel immediately.

Stops scheduling new requests once ``--max-usd`` (default 5) is spent. A few already started can finish past it.
Writes ``runs/truncation_replay/REPORT.md`` and ``data.json``. No GPU.

    uv run python -m experiments.truncation_replay --max-usd 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from taic.llm import CostLedger, LLM, LLMError, Transcript, load_config
from taic.orchestrator import _retry_without_reasoning
from taic.parse import extract_metal

REPO = Path(__file__).resolve().parent.parent
ARMS = ("as_sent", "no_reasoning", "low_effort", "write_now", "off_effort")
KERNELS = {
    "fa_paper": "flash_attention_kernel",
    "conv2d_paper": "conv2d_kernel",
    "mamba2": "mamba2_chunk_scan_kernel",
}
WRITE_NOW = (
    "Your previous reply was cut off before it contained a kernel. Do not repeat that reasoning. "
    "Reply now with the complete kernel in exactly one ```metal block."
)


def _samples(n: int) -> list[dict]:
    found = []
    for name, kernel in KERNELS.items():
        path = REPO / "runs" / "p1" / name / "transcript.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            if doc.get("event") != "chat" or doc.get("finish_reason") != "length":
                continue
            content = (doc.get("reply") or {}).get("content") or ""
            if "kernel void" in content or "```metal" in content:
                continue
            messages = doc.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            found.append({"run": name, "label": str(doc.get("label")), "messages": messages, "kernel": kernel})
    random.Random(0).shuffle(found)
    return found[:n]


def _messages(arm: str, messages: list[dict]) -> list[dict]:
    if arm == "as_sent":
        return [dict(m) for m in messages]
    out = _retry_without_reasoning(messages)
    if arm == "write_now":
        out = out + [{"role": "user", "content": WRITE_NOW}]
    return out


async def _one(llm: LLM, arm: str, sample: dict) -> dict:
    try:
        reply = await llm.chat(_messages(arm, sample["messages"]), label=f"{sample['run']}:{sample['label']}:{arm}")
    except LLMError as e:
        return {"run": sample["run"], "label": sample["label"], "arm": arm, "finish_reason": "error", "cost_usd": 0.0, "content_chars": 0, "has_kernel": False, "error": str(e)[:200]}
    text = reply.text()
    return {
        "run": sample["run"], "label": sample["label"], "arm": arm, "finish_reason": reply.finish_reason,
        "cost_usd": reply.cost_usd, "content_chars": len(text), "has_kernel": extract_metal(text, sample["kernel"]) is not None,
    }


def _report(rows: list[dict], spent: float) -> str:
    lines = ["# Truncation replay", "", f"Spent ${spent:.3f} on {len(rows)} requests.", ""]
    lines += ["| arm | requests | kernels | truncated again | mean cost |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        own = [r for r in rows if r["arm"] == arm]
        if not own:
            continue
        kernels = sum(1 for r in own if r["has_kernel"])
        again = sum(1 for r in own if r["finish_reason"] == "length")
        mean = sum(r["cost_usd"] for r in own) / len(own)
        lines.append(f"| {arm} | {len(own)} | {kernels} | {again} | ${mean:.4f} |")
    lines += [
        "",
        "The search retries with `no_reasoning`: drop the echoed reasoning and the empty assistant turn, ask once more, and do not spend a repair turn.",
        "If another arm produces more kernels for less money, change `_retry_without_reasoning` to that arm before reading the A/B.",
    ]
    return "\n".join(lines) + "\n"


async def replay(*, n: int, max_usd: float, out: Path, concurrency: int) -> int:
    cfg, _ = load_config(None, preset="base")
    out.mkdir(parents=True, exist_ok=True)
    from dataclasses import replace

    ledger = CostLedger(cfg, max_usd)
    normal = LLM(cfg, ledger, Transcript(out / "transcript.jsonl"), max_concurrency=concurrency)
    low = LLM(replace(cfg, reasoning_effort="low"), ledger, normal.transcript, max_concurrency=concurrency)
    off = LLM(replace(cfg, reasoning_effort="off"), ledger, normal.transcript, max_concurrency=concurrency)
    samples = _samples(n)
    if not samples:
        print("no truncated replies found")
        return 1
    jobs = [(sample, arm) for sample in samples for arm in ARMS]
    rows: list[dict] = []

    async def worker() -> None:
        while jobs and ledger.total_usd < max_usd:
            sample, arm = jobs.pop(0)
            if ledger.total_usd >= max_usd:
                return
            print(f"replay {sample['run']} {sample['label']} {arm} spent ${ledger.total_usd:.3f}", flush=True)
            rows.append(await _one(off if arm == "off_effort" else low if arm == "low_effort" else normal, arm, sample))

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    (out / "data.json").write_text(json.dumps(rows, indent=1))
    text = _report(rows, ledger.total_usd)
    (out / "REPORT.md").write_text(text)
    print(text)
    await normal.aclose()
    await low.aclose()
    await off.aclose()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="runs/truncation_replay")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args(argv)
    return asyncio.run(replay(n=a.n, max_usd=a.max_usd, out=REPO / a.out, concurrency=a.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
