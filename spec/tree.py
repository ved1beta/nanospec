"""Static draft tree: topk nodes at each of `depth` levels (EAGLE-2 selection: the
global top-k children by cumulative log-prob at every level). N = topk * depth nodes.

Node i has token[i], parent[i] (-1 = the root, i.e. the last committed token) and
depth[i] = 1 + depth[parent]. Verify rows are [root] + nodes; row r attends the prefix,
the root, and its ancestors-or-self. Retrieval: walk parents from any node to the root.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Tree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]

    @property
    def n(self) -> int:
        return len(self.tokens)

    def ancestors(self, i: int) -> list[int]:
        """Ancestors-or-self of node i, root-side first."""
        path = []
        while i >= 0:
            path.append(i)
            i = self.parents[i]
        return path[::-1]


def ancestor_matrix(parents: list[int], device=None) -> torch.Tensor:
    """[N, N] bool: A[i, j] = j is an ancestor-or-self of i. Built on the CPU (N <= 32,
    a Python loop beats N tiny GPU kernels) and moved once if a device is given."""
    n = len(parents)
    a = torch.eye(n, dtype=torch.bool)
    for i, p in enumerate(parents):
        if p >= 0:
            a[i] |= a[p]  # parents precede children, so a[p] is complete
    return a.to(device) if device is not None else a


def verify_mask(prefix_len: int, parents: list[int], device) -> torch.Tensor:
    """[1+N, prefix_len+1+N] bool for rows [root] + nodes over kv = prefix + root + nodes."""
    n = len(parents)
    m = torch.zeros(1 + n, prefix_len + 1 + n, dtype=torch.bool)
    m[:, : prefix_len + 1] = True  # everyone sees the prefix and the root
    m[1:, prefix_len + 1 :] = ancestor_matrix(parents)
    return m.to(device, non_blocking=True)


def draft_mask(prefix_len: int, parents: list[int], rows: list[int], n_slots: int, device) -> torch.Tensor:
    """[len(rows), prefix_len+n_slots] bool for draft rows = nodes `rows` (as inputs) over
    kv = draft prefix + the tree's slot region: prefix + ancestors-or-self."""
    a = ancestor_matrix(parents)
    m = torch.zeros(len(rows), prefix_len + n_slots, dtype=torch.bool)
    m[:, :prefix_len] = True
    m[:, prefix_len : prefix_len + a.shape[0]] = a[rows]
    return m.to(device, non_blocking=True)


def longest_accepted(tree: Tree, argmax: list[int]) -> tuple[list[int], int]:
    """Greedy acceptance. argmax[0] is the target's choice at the root row, argmax[1+i]
    at node i. Returns (accepted node indices root->leaf, bonus token)."""
    children: dict[int, list[int]] = {}
    for i, p in enumerate(tree.parents):
        children.setdefault(p, []).append(i)
    path, cur, want = [], -1, argmax[0]
    while True:
        nxt = next((c for c in children.get(cur, []) if tree.tokens[c] == want), None)
        if nxt is None:
            return path, want
        path.append(nxt)
        cur, want = nxt, argmax[1 + nxt]


def select_children(
    scores: torch.Tensor, cand_tokens: torch.Tensor, cand_parents: list[int], topk: int
) -> tuple[list[int], list[int], torch.Tensor]:
    """scores [C] cumulative log-probs of C candidate children (parent index per
    candidate). Keeps the global top-k: returns (tokens, parents, scores) of the kept."""
    k = min(topk, scores.numel())
    best = scores.topk(k).indices
    return cand_tokens[best].tolist(), [cand_parents[i] for i in best.tolist()], scores[best]
