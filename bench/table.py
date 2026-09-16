"""Fold bench JSON lines into the README table (median over runs)."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main(paths: list[str]) -> None:
    rows = [json.loads(l) for p in paths for l in open(p) if l.startswith("{")]
    by = defaultdict(list)
    for r in rows:
        by[(r["engine"], r["config"], r["bs"])].append(r)
    print("| engine | config | bs | out tok/s | TPOT ms | accepted len | runs | commit/version |")
    print("|---|---|---|---|---|---|---|---|")
    for (e, c, bs), rs in sorted(by.items()):
        med = lambda k: statistics.median(x[k] for x in rs if x[k] is not None) if any(x[k] is not None for x in rs) else "-"
        ver = rs[0].get("vllm") or rs[0].get("sglang") or rs[0].get("commit")
        print(f"| {e} | {c} | {bs} | {med('tok_s')} | {med('tpot_ms')} | {med('accepted_len')} | {len(rs)} | {ver} |")
    print(f"\nGPU: {rows[0]['gpu']}" if rows else "")


if __name__ == "__main__":
    main(sys.argv[1:])
