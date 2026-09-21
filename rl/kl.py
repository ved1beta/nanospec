"""Per-token divergence of the power-rule sampler from the policy, closed form and measured.

At a row where an argmax draft d with target probability p is tested, the sampler emits d
with probability p**alpha and otherwise a token from the renormalised remainder, so

    KL(pi || mu) = p (1 - alpha) log p + (1 - p) log[(1 - p) / (1 - p**alpha)]

which is 0 at alpha = 1, bounded by a function of alpha alone for alpha > 0 (worst case
over p: 0.0035 nats at 0.9, 0.038 at 0.7, 0.136 at 0.5, 0.37 at 0.3), and diverges as
alpha -> 0 (the deterministic tau / top-k rule: mu has no support off the draft). Rows
without a tested draft (bonus rows) contribute 0.

    python -m rl.kl --alpha 0.3 telemetry.jsonl      # measured mean per emitted token, per step
    python -m rl.kl --table                          # the worst-case bound per alpha
"""

from __future__ import annotations

import argparse
import json
import math


def kl_row(p: float, alpha: float) -> float:
    if p <= 0.0 or p >= 1.0 or alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return math.inf
    return p * (1 - alpha) * math.log(p) + (1 - p) * math.log((1 - p) / (1 - p**alpha))


def worst_case(alpha: float, grid: int = 20000) -> tuple[float, float]:
    best = max(((kl_row(i / grid, alpha), i / grid) for i in range(1, grid)), key=lambda t: t[0])
    return best


def measure(path: str, alpha: float, chain_only: bool = True):
    """Stream a telemetry file -> {step: (sum of row KLs, emitted tokens, tested drafts)}."""
    acc: dict[int, list[float]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if chain_only and r.get("topk", 1) != 1:
                raise ValueError("the closed form is for chain (argmax, topk=1) proposals")
            s = acc.setdefault(r["step"], [0.0, 0, 0])
            for nd in r["nodes"]:
                if nd["d"] != "-":  # tested at its row: accepted (walk continues) or rejected (resample there)
                    s[0] += kl_row(nd["p"], alpha)
                    s[2] += 1
            s[1] += len(r.get("emitted", [])) or (r["accepted"] + 1)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry", nargs="?")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.table or not a.telemetry:
        for al in (0.9, 0.7, 0.5, 0.3, 0.1):
            k, p = worst_case(al)
            print(f"alpha {al}: sup_p KL(pi||mu) = {k:.4f} nats at p = {p:.3f}")
        return
    acc = measure(a.telemetry, a.alpha)
    steps = sorted(acc)
    per = {s: acc[s][0] / max(acc[s][1], 1) for s in steps}
    tested = {s: acc[s][2] / max(acc[s][1], 1) for s in steps}
    tot = sum(acc[s][0] for s in steps) / max(sum(acc[s][1] for s in steps), 1)
    print(json.dumps({"alpha": a.alpha, "mean_kl_pi_mu_per_token": tot, "first20": sum(per[s] for s in steps[:20]) / max(len(steps[:20]), 1),
                      "last20": sum(per[s] for s in steps[-20:]) / max(len(steps[-20:]), 1), "tested_per_token": sum(tested.values()) / len(steps),
                      "steps": len(steps)}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"alpha": a.alpha, "per_step": per, "tested_per_token": tested}, f)


if __name__ == "__main__":
    main()
