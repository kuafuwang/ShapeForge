"""Measure the failure fingerprints on the two documented corpora.

``corpus_disk``  = ``runs/**/candidates/*/report.json`` (the same glob the candidate reader
                   sees, excluding ``agent_work/*/report.json`` and the two root-level sim
                   reports by construction).
``corpus_index`` = the ``candidates`` table after an ordinary ingest of ``runs/``.

The plan's numbers (288 failures / 22 signatures / 92.01% on disk; 285 / 22 / 91.93% in the
index) come from this script. ``observe.failure_sig`` is the authority: when the module
changes, rerun the script and update the baseline, never the other way round.

Usage::

    uv run python -m experiments.failure_corpus
    uv run python -m experiments.failure_corpus --runs runs --json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import tempfile
from pathlib import Path

from observe.failure_sig import signature

_FAILED = ("", "ok")


def _disk_reports(runs: Path) -> list[Path]:
    """@brief Every candidate report under ``runs/``, using the reader's glob.

    @param runs Runs root.
    @return Sorted report paths.
    """
    return sorted(runs.glob("**/candidates/*/report.json"))


def _summary(failures: list[tuple[str, str, str]]) -> dict:
    """@brief Count signatures and top-6 coverage for ``(status, stage, message)`` triples.

    @param failures Failing candidates.
    @return Dict with totals, signature count and the top clusters.
    """
    if not failures:
        return {"failed": 0, "distinct_messages": 0, "signatures": 0, "top6": 0, "coverage": 0.0, "clusters": []}
    signatures = collections.Counter(signature(status, stage, message) for status, stage, message in failures)
    messages = {str(message) for _status, _stage, message in failures}
    top = signatures.most_common(6)
    covered = sum(count for _sig, count in top)
    return {
        "failed": len(failures),
        "distinct_messages": len(messages),
        "signatures": len(signatures),
        "top6": covered,
        "coverage": round(covered / len(failures) * 100, 2),
        "clusters": [{"sig": sig, "n": count} for sig, count in signatures.most_common()],
    }


def corpus_disk(runs: Path) -> dict:
    """@brief Measure the on-disk candidate reports.

    @param runs Runs root.
    @return Report dict.
    """
    files = _disk_reports(runs)
    failures = []
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        status = str(document.get("status") or "")
        if status in _FAILED:
            continue
        failures.append((status, str(document.get("stage") or ""), document.get("message")))
    return {"reports": len(files), **_summary(failures)}


def corpus_index(runs: Path) -> dict:
    """@brief Ingest ``runs/`` into a throwaway database and measure the ``candidates`` rows.

    @param runs Runs root.
    @return Report dict.
    """
    from observe.index import Index

    with tempfile.TemporaryDirectory(prefix="taic-failure-corpus-") as tmp:
        root = Path(tmp)
        os.environ["AI_LOWERLING_RUNS_DIR"] = str(runs.resolve())
        os.environ["OBSERVE_DB"] = str(root / "observe.sqlite")
        os.environ["AI_LOWERLING_BOOKS"] = str(root / "none.sqlite")
        os.environ["OBSERVE_CARRYOVER"] = str(root / "carryover.json")
        os.environ["OBSERVE_TICK_INTERVAL"] = "0"
        os.environ["OBSERVE_FULL_TICK_INTERVAL"] = "3600"
        import observe.index as index_mod

        index_mod._SINGLE = None
        index = Index(root / "observe.sqlite")
        index.sync_known()
        rows = index.conn.execute(
            "SELECT status, stage, message, failure_sig FROM candidates"
        ).fetchall()
        failures = [
            (str(row["status"] or ""), str(row["stage"] or ""), row["message"])
            for row in rows if str(row["status"] or "") not in _FAILED
        ]
        stored = index.conn.execute(
            "SELECT sig, COUNT(*) AS n FROM failure_members GROUP BY sig ORDER BY n DESC, sig"
        ).fetchall()
        result = {
            "candidates": len(rows),
            **_summary(failures),
            "members": sum(int(row["n"]) for row in stored),
            "member_signatures": len(stored),
            "top6_members": sum(int(row["n"]) for row in stored[:6]),
        }
        index.conn.close()
        return result


def main(argv: list[str] | None = None) -> int:
    """@brief CLI: print both corpora as JSON.

    @param argv Optional argv (tests pass none).
    @return Process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m experiments.failure_corpus")
    parser.add_argument("--runs", default="runs", help="runs/ root")
    parser.add_argument("--json", action="store_true", help="print only the JSON document")
    args = parser.parse_args(argv)
    runs = Path(args.runs)
    disk = corpus_disk(runs)
    from_index = corpus_index(runs)
    document = {"corpus_disk": disk, "corpus_index": from_index}
    print(json.dumps(document, ensure_ascii=False, indent=2))
    if not args.json:
        print(
            f"disk  {disk['failed']}/{disk['reports']} 失败, {disk['distinct_messages']} 种原始消息, "
            f"{disk['signatures']} 签名, 前6 {disk['top6']}={disk['coverage']}%"
        )
        print(
            f"index {from_index['failed']}/{from_index['candidates']} 失败, "
            f"{from_index['signatures']} 签名, 前6 {from_index['top6']}={from_index['coverage']}%, "
            f"成员表 {from_index['members']} 行 / {from_index['member_signatures']} 簇"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
