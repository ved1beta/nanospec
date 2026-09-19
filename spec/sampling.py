"""Sampling and acceptance as pure functions.

process:  logits -> log-probs under (temperature, top-k, top-p), one triple per row.
draw:     one token per row from the residual max(0, p - q), inverse-CDF on a uniform.
walk:     the host-side acceptance over one request's tree, sibling by sibling.

Verification is Leviathan's test with q the proposal at the node's parent row. Argmax
drafts are point masses, so the test is `u < p(d)`, and the residual after rejecting
siblings c1..ck is p with them removed and renormalised: the emitted marginal is exactly p,
for chain and tree alike. Greedy rows (T = 0) give p in {0, 1}: accept iff d is the argmax.

behavior_logprob: log mu(a | s), the probability the sampler actually emitted a, for the
trainer's importance weight pi(a | s) / mu(a | s) (relaxed and power acceptance are
off-policy samplers).
"""

from __future__ import annotations

import math

import torch


def process(z: torch.Tensor, temp: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor, nucleus: bool = True) -> torch.Tensor:
    """z [c, V] fp32 raw logits, per-row params -> processed log-probs (-inf outside the
    nucleus; `nucleus` False skips the sort when no row uses top-k / top-p). Greedy rows
    (temp 0) are scaled by 1 and left to the caller."""
    z = z / temp.where(temp > 0, 1.0)[:, None]
    if nucleus:
        srt, idx = z.sort(-1, descending=True)
        p = srt.softmax(-1)
        keep = (p.cumsum(-1) - p) <= top_p[:, None]  # tokens up to and including the one that crosses top_p
        keep &= (torch.arange(z.shape[-1], device=z.device) < top_k[:, None]) | (top_k[:, None] <= 0)
        z = torch.full_like(z, float("-inf")).scatter_(-1, idx, srt.masked_fill(~keep, float("-inf")))
    return z.log_softmax(-1)


def draw(lp: torch.Tensor, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """lp [c, V] processed log-probs, q [c, V] proposal mass already tested at this row,
    u [c] uniforms -> token per row from norm(max(0, p - q)); p itself when q is empty."""
    cdf = (lp.exp() - q).clamp_min(0).cumsum(-1)
    return torch.searchsorted(cdf, (u * cdf[:, -1])[:, None], right=True).squeeze(1).clamp_max(lp.shape[-1] - 1)


def walk(children: dict[int, list[int]], p: list[float], q: list[float], u: list[float], rank: list[int] | None,
         relaxed: tuple[float, int] | None, alpha: float = 1.0) -> tuple[list[int], list[str], list[float]]:
    """children[node] (node -1 = root), per node: p = target prob of its token at its
    parent row, q = the proposal's prob of it, u = its uniform. Exact rule: test each
    sibling against the residual left by the ones rejected before it, accept with
    probability min(1, p / (q * rest)). Power rule (alpha < 1): that probability to the
    power alpha -- more accepting, still full support. Relaxed (tau, top-k): accept iff
    p >= tau or the token ranks in the target's top-k (deterministic).
    -> (accepted path root->leaf, decision per node: accept|reject|-, accept prob per node)."""
    path, cur, dec, acc = [], -1, ["-"] * len(p), [0.0] * len(p)
    while True:
        rest, nxt = 1.0, None
        for c in children.get(cur, []):
            if relaxed is not None:
                a = float(p[c] >= relaxed[0] or (rank is not None and rank[c] < relaxed[1]))
            else:
                a = min(1.0, p[c] / (q[c] * rest)) ** alpha if q[c] * rest > 0 else float(p[c] > 0)  # one-hot q -> min(1, p / rest)
            acc[c] = a
            ok = a >= 1.0 or u[c] < a
            dec[c] = "accept" if ok else "reject"
            if ok:
                nxt = c
                break
            rest = max(rest - p[c], 0.0)
        if nxt is None:
            return path, dec, acc
        path.append(nxt)
        cur = nxt


def behavior_logprob(source: str, sample_logprob: float, rejected: list[tuple[float, float]], accept_prob: float, exact: bool) -> float:
    """log mu(a | s) of an emitted token: the probability the sampler emitted it given the
    realised history. Exact acceptance: the emitted marginal at every position is the
    target's processed distribution, so mu = p (sample_logprob) whatever the source.
    Otherwise the walk's path probability: every sibling tested and rejected before it
    contributes (1 - accept_prob), then the accepted draft's own accept_prob, or for a
    resample row p(a) / (1 - sum p(rejected)), or p(a) at a bonus row. Under the
    deterministic relaxed rule accept probs are 0 / 1, so mu = 1 on an accepted draft and
    the sampler has NO support on the target's other tokens there: the weights are exact
    on the sampler's support, but reweighting cannot recover the missing mass
    (tests/test_behavior.py measures it). The power rule keeps full support."""
    if exact or source == "bonus" and not rejected:
        return sample_logprob
    lp = sum(math.log(max(1.0 - a, 1e-300)) for a, _ in rejected)
    if source == "draft":
        return lp + math.log(max(accept_prob, 1e-300))
    if source == "resample":
        return lp + sample_logprob - math.log(max(1.0 - sum(pp for _, pp in rejected), 1e-300))
    return lp + sample_logprob
