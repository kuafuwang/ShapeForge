"""Re-send truncated replies that contained no kernel, four ways, and compare which one produces a kernel.

The four arms, each a fresh request built from the original messages:

* ``as_sent``       the messages exactly as they were sent (the control);
* ``no_reasoning``  ``reasoning_content`` removed, and an empty truncated assistant turn dropped;
* ``low_effort``    ``no_reasoning``, with ``reasoning_effort`` set to ``low``;
* ``off_effort``    ``no_reasoning``, with ``reasoning_effort`` set to ``off``;
* ``write_now``     ``no_reasoning`` plus one line telling the model to write the kernel immediately.

Stops scheduling new requests once ``--max-usd`` (default 5) is spent. A few already started can finish past it.
Writes ``runs/apple-m4-max/truncation_replay/REPORT.md`` and ``data.json``. No GPU.

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
    """@brief 从已完成的 transcript 里抽出没写出内核就被截断的请求。

    @param n 最多取多少个样本。
    @return 打乱后的样本列表，每项带原消息与要抽的 kernel 名。
    """
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
    """@brief 按实验组改造原消息：删推理、降 effort 或加一句催写法。

    @param arm 实验组名，取 ``ARMS`` 里的值。
    @param messages 原始请求消息。
    @return 重新拼好的消息列表。
    """
    if arm == "as_sent":
        return [dict(m) for m in messages]
    out = _retry_without_reasoning(messages)
    if arm == "write_now":
        out = out + [{"role": "user", "content": WRITE_NOW}]
    return out


async def _one(llm: LLM, arm: str, sample: dict) -> dict:
    """@brief 重发一次请求，并记下是否抽到了内核、是否又被截断。

    @param llm 要用的 LLM 客户端。
    @param arm 实验组名。
    @param sample ``_samples`` 返回的样本。
    @return 一行结果：结束原因、花费、字数与是否含内核；调用出错时 ``finish_reason`` 为 ``error``。
    """
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
    """@brief 按实验组统计请求数、成功数、再次截断数与平均花费。

    @param rows 每次重发的结果行。
    @param spent 本次总共花了多少美元。
    @return markdown 报告文本。
    """
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
    """@brief 并发重发截断请求，写出 ``data.json`` 与 ``REPORT.md``。

    花费达到 ``max_usd`` 后不再发新请求，已经开始的那几个可以超出。

    @param n 最多重发多少个样本。
    @param max_usd 本次的花费上限，单位美元。
    @param out 输出目录。
    @param concurrency 并发请求数。
    @return 进程退出码；没有找到被截断的回复时返回 1。
    """
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
    """@brief ``python -m experiments.truncation_replay`` 的入口。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="runs/apple-m4-max/truncation_replay")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args(argv)
    return asyncio.run(replay(n=a.n, max_usd=a.max_usd, out=REPO / a.out, concurrency=a.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
