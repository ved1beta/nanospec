"""Static draft tree: topk nodes at each of `depth` levels (EAGLE-2 selection: the
global top-k children by cumulative log-prob at every level). N = topk * depth nodes.
A chain is the tree with linear parents.

Node i has token[i], parent[i] (-1 = the root, i.e. the last committed token) and
depth[i] = 1 + depth[parent]. Verify rows are [root] + nodes; row r attends the prefix,
the root, and its ancestors-or-self. Node i is scored by row 1 + parent[i] (the root row
for parent -1); acceptance is spec/sampling.walk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Tree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    q: list[float] | None = None  # the draft's log-prob of each node's token

    @classmethod
    def chain(cls, tokens: list[int], q: list[float] | None = None) -> "Tree":
        return cls(tokens, list(range(-1, len(tokens) - 1)), list(range(1, len(tokens) + 1)), q)

    @property
    def n(self) -> int:
        return len(self.tokens)

    def children(self) -> dict[int, list[int]]:
        """node -> children in node order; -1 is the root."""
        out: dict[int, list[int]] = {}
        for i, p in enumerate(self.parents):
            out.setdefault(p, []).append(i)
        return out

    def ancestors(self, i: int) -> list[int]:
        """Ancestors-or-self of node i, root-side first."""
        path = []
        while i >= 0:
            path.append(i)
            i = self.parents[i]
        return path[::-1]


def ancestors(parents: torch.Tensor, depth: int) -> torch.Tensor:
    """parents [B, N] (-1 = root) -> A [B, N, N] bool: A[b, i, j] = j is an ancestor-or-self
    of i. depth-1 pointer-chasing steps cover every node at depth <= depth."""
    B, N = parents.shape
    b, i = torch.arange(B, device=parents.device)[:, None], torch.arange(N, device=parents.device)[None, :]
    A = torch.eye(N, dtype=torch.bool, device=parents.device).expand(B, N, N).clone()
    anc = parents
    for _ in range(depth - 1):
        A[b, i, anc.clamp_min(0)] |= anc >= 0
        anc = parents.gather(1, anc.clamp_min(0)).where(anc >= 0, anc)
    return A


def flat_masks(prefix: list[int], rows: torch.Tensor, extra: int) -> torch.Tensor:
    """Per request b an [R, prefix[b] + extra + N] block: True over the first
    prefix[b] + extra columns (the prefix, and the root for a verify), then rows[b]
    ([B, R, N] bool over the tree slots); flattened and concatenated (FlashInfer's
    custom_mask layout). Built on the device with no per-request launches."""
    B, R, N = rows.shape
    dev = rows.device
    L = torch.tensor(prefix, device=dev) + extra
    W = L + N
    total = sum(R * (p + extra + N) for p in prefix)
    b = torch.repeat_interleave(torch.arange(B, device=dev), R * W, output_size=total)
    local = torch.arange(total, device=dev) - torch.cat([W.new_zeros(1), (R * W).cumsum(0)])[b]
    r, j = local // W[b], local % W[b] - L[b]
    return (j < 0) | rows[b, r, j.clamp_min(0)]
