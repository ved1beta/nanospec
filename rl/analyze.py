"""Pull GRPO run logs from the Modal volume and draw the headline figures.

    python -m rl.analyze --fetch e0-exact-none e1-a0.7-none e2-a0.7-exact ...   # -> runs/<name>/log.jsonl
    python -m rl.analyze                                                         # figures from runs/*/log.jsonl

Figure 1: reward vs step per run.  Figure 2: accepted drafts / step and rollout tok/s vs step.
Figure 3: the knee: final reward (mean of the last 20 steps) vs rollout tok/s, one point per run,
          corrected and uncorrected joined per alpha.  Table: one row per run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

RUNS = Path("runs")


def fetch(names: list[str]) -> None:
    for n in names:
        (RUNS / n).mkdir(parents=True, exist_ok=True)
        for f in ("log.jsonl", "args.json"):
            subprocess.run(["modal", "volume", "get", "nanospec-runs", f"{n}/{f}", str(RUNS / n / f), "--force"], check=False, capture_output=True)


def load() -> dict[str, tuple[dict, list[dict]]]:
    out = {}
    for d in sorted(RUNS.iterdir()):
        log = d / "log.jsonl"
        if log.exists() and log.stat().st_size:
            args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
            out[d.name] = (args, [json.loads(l) for l in log.read_text().splitlines()])
    return out


def label(args: dict) -> str:
    acc = args.get("acceptance", "?")
    a = "exact" if acc == "exact" else f"alpha={args.get('alpha')}" if acc == "power" else f"tau={args.get('tau')}/top{args.get('accept_topk')}"
    return f"{a}, {args.get('correction', '?')}"


def tail_mean(rows, key, n=20):
    v = [r[key] for r in rows[-n:] if key in r]
    return sum(v) / len(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", nargs="*")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.fetch)
    runs = load()
    if not runs:
        print("no runs")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'run':22s} {'setting':22s} {'steps':>5s} {'reward':>7s} {'eval':>6s} {'acc':>5s} {'tok/s':>6s} {'ess':>5s} {'kl(mu|pi)':>9s} {'kl(pi|pi0)':>10s}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    knee = {}
    for name, (a, rows) in runs.items():
        s = [r["step"] for r in rows]
        lab = label(a)
        ax[0].plot(s, [r["reward"] for r in rows], label=lab, lw=1)
        ax[1].plot(s, [r["accepted"] for r in rows], label=lab, lw=1)
        ax[2].plot(s, [r["tok_s"] for r in rows], label=lab, lw=1)
        ev = [r["eval_acc"] for r in rows if "eval_acc" in r]
        rec = dict(reward=tail_mean(rows, "reward"), tok_s=tail_mean(rows, "tok_s"), acc=tail_mean(rows, "accepted"), ess=tail_mean(rows, "ess"),
                   kl=tail_mean(rows, "kl_mu_pi"), kl0=tail_mean(rows, "kl_pi_pi0"), eval=ev[-1] if ev else float("nan"), n=len(rows))
        knee[name] = (a, rec)
        print(f"{name:22s} {lab:22s} {rec['n']:5d} {rec['reward']:7.3f} {rec['eval']:6.3f} {rec['acc']:5.2f} {rec['tok_s']:6.0f} {rec['ess']:5.2f} {rec['kl']:9.4f} {rec['kl0']:10.4f}")
    for x, t in zip(ax, ("GSM8K reward on rollouts", "accepted drafts / step", "rollout output tok/s")):
        x.set_title(t)
        x.set_xlabel("GRPO step")
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)

    fig, x = plt.subplots(figsize=(5.5, 4.2))
    for name, (a, rec) in knee.items():
        c = "tab:red" if a.get("correction") == "exact" else "tab:blue"
        x.scatter(rec["tok_s"], rec["reward"], color=c)
        x.annotate(label(a), (rec["tok_s"], rec["reward"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    by_alpha: dict[float, dict[str, tuple]] = {}
    for name, (a, rec) in knee.items():
        by_alpha.setdefault(a.get("alpha", 1.0), {})[a.get("correction")] = (rec["tok_s"], rec["reward"])
    for al, d in by_alpha.items():
        if "none" in d and "exact" in d:
            x.annotate("", xy=d["exact"], xytext=d["none"], arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    x.set_xlabel("rollout output tok/s (mean of last 20 steps)")
    x.set_ylabel("reward (mean of last 20 steps)")
    x.set_title("speed / quality; blue: no correction, red: exact IS")
    x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "knee.png", dpi=150)
    print(f"wrote {out}/curves.png, {out}/knee.png")


if __name__ == "__main__":
    main()
