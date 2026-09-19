"""Sampling and acceptance as pure functions.

process:  logits -> log-probs under (temperature, top-k, top-p), one triple per row.
draw:     one token per row from the residual max(0, p - q), inverse-CDF on a uniform.
walk:     the host-side acceptance over one request's tree, sibling by sibling.

Verification is Leviathan's test with q the proposal at the node's parent row. Argmax
drafts are point masses, so the test is `u < p(d)`, and the residual after rejecting
siblings c1..ck is p with them removed and renormalised: the emitted marginal is exactly p,
for chain and tree alike. Greedy rows (T = 0) give p in {0, 1}: accept iff d is the argmax.
"""

from __future__ import annotations

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
         relaxed: tuple[float, int] | None) -> tuple[list[int], list[str]]:
    """children[node] (node -1 = root), per node: p = target prob of its token at its
    parent row, q = the proposal's prob of it, u = its uniform. Exact rule: test each
    sibling against the residual left by the ones rejected before it, min(1, p/q) for a
    sampled proposal. Relaxed (tau, top-k): accept if p >= tau or the token ranks in the
    target's top-k. -> (accepted path root->leaf, decision per node: accept|reject|-)."""
    path, cur, dec = [], -1, ["-"] * len(p)
    while True:
        rest, nxt = 1.0, None
        for c in children.get(cur, []):
            if relaxed is not None:
                ok = p[c] >= relaxed[0] or (rank is not None and rank[c] < relaxed[1])
            else:
                ok = u[c] * rest * q[c] < p[c]  # u < min(1, p / (q * rest)); one-hot q -> u < p / rest
            dec[c] = "accept" if ok else "reject"
            if ok:
                nxt = c
                break
            rest = max(rest - p[c], 0.0)
        if nxt is None:
            return path, dec
        path.append(nxt)
        cur = nxt
