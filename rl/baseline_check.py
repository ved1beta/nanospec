"""Objection 2 check: per group, the plain mean reward vs the self-normalised IS-weighted
mean with clipped sequence ratios, over training. Reads <run>/seqs.jsonl (rl/grpo.py).

    python -m rl.baseline_check runs/<run> [--group 8] [--clip 5.0]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--clip", type=float, default=5.0, help="clip |sum_t log w_t| at this value before exponentiating")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (Path(a.run) / "seqs.jsonl").read_text().splitlines()]
    gaps, rel = [], []
    for r in rows:
        R, L = r["reward"], r["logw"]
        g = 0.0
        for s in range(0, len(R), a.group):
            rr, lw = R[s : s + a.group], L[s : s + a.group]
            w = [math.exp(max(-a.clip, min(a.clip, x))) for x in lw]
            plain = sum(rr) / len(rr)
            snis = sum(wi * ri for wi, ri in zip(w, rr)) / max(sum(w), 1e-12)
            g += abs(snis - plain)
        gaps.append(g / (len(R) / a.group))
        rel.append(gaps[-1] / max(sum(R) / len(R), 1e-6))
    n = len(gaps)
    print(json.dumps({"steps": n, "mean_abs_gap": sum(gaps) / n, "first20": sum(gaps[:20]) / max(len(gaps[:20]), 1),
                      "last20": sum(gaps[-20:]) / max(len(gaps[-20:]), 1), "mean_gap_over_mean_reward": sum(rel) / n}))


if __name__ == "__main__":
    main()
