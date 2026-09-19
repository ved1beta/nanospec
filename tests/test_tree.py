import torch

from spec.sampling import walk
from spec.tree import Tree, ancestors, flat_masks


def _tree():
    # depth1: n0 n1 ; depth2: n2(p=n0) n3(p=n1) ; depth3: n4(p=n2) n5(p=n2)
    return Tree(tokens=[10, 11, 20, 21, 30, 31], parents=[-1, -1, 0, 1, 2, 2], depths=[1, 1, 2, 2, 3, 3])


def test_ancestors_and_masks():
    t = _tree()
    par = torch.tensor([t.parents, t.parents])
    a = ancestors(par, 3)
    assert a[0, 4].tolist() == [True, False, True, False, True, False]
    assert a[1, 3].tolist() == [False, True, False, True, False, False]
    assert (ancestors(par, 2)[0, 4] == torch.tensor([False, False, True, False, True, False])).all()  # one hop short
    # verify blocks: rows [root] + nodes over prefix + root + 6 slots, prefixes 3 and 5
    rows = torch.cat([torch.zeros_like(a[:, :1]), a], 1)
    m = flat_masks([3, 5], rows, 1)
    assert m.numel() == 7 * 10 + 7 * 12
    m0, m1 = m[:70].view(7, 10), m[70:].view(7, 12)
    assert m0[:, :4].all() and m1[:, :6].all()  # prefix + root
    assert m0[0, 4:].tolist() == [False] * 6 and m1[5, 6:].tolist() == a[1, 4].tolist()
    # draft level block: rows = nodes 2, 3 over prefix + 6 slots, no root column
    d = flat_masks([5], a[:1, 2:4], 0).view(2, 11)
    assert d[:, :5].all() and d[0, 5:].tolist() == a[0, 2].tolist()


def _greedy_walk(t, argmax):
    """Greedy acceptance through walk: p is 1 iff the node's token is the argmax at its
    parent row; the bonus is the argmax at the last accepted node's row."""
    p = [float(argmax[1 + t.parents[i]] == t.tokens[i]) for i in range(t.n)]
    path, _ = walk(t.children(), p, [1.0] * t.n, [0.5] * t.n, None, None)
    return path, argmax[1 + path[-1] if path else 0]


def test_greedy_walk():
    t = _tree()
    # target says: root->10, n0->20, n2->31, n5->99  => path n0, n2, n5 ; bonus 99
    assert _greedy_walk(t, [10, 20, 0, 31, 0, 0, 99]) == ([0, 2, 5], 99)
    # root picks 11, n1 picks something not offered => path n1, bonus 77
    assert _greedy_walk(t, [11, 0, 77, 0, 0, 0, 0]) == ([1], 77)
    # root rejects everything
    assert _greedy_walk(t, [5, 0, 0, 0, 0, 0, 0]) == ([], 5)
    assert t.ancestors(5) == [0, 2, 5]
    assert Tree.chain([1, 2, 3]).parents == [-1, 0, 1]
