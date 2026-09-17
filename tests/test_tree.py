import torch

from spec.tree import Tree, ancestor_matrix, draft_mask, longest_accepted, verify_mask


def _tree():
    # depth1: n0 n1 ; depth2: n2(p=n0) n3(p=n1) ; depth3: n4(p=n2) n5(p=n2)
    return Tree(tokens=[10, 11, 20, 21, 30, 31], parents=[-1, -1, 0, 1, 2, 2], depths=[1, 1, 2, 2, 3, 3])


def test_ancestor_matrix_and_masks():
    t = _tree()
    a = ancestor_matrix(t.parents, "cpu")
    assert a[4].tolist() == [True, False, True, False, True, False]
    assert a[3].tolist() == [False, True, False, True, False, False]
    m = verify_mask(3, t.parents, "cpu")
    assert m.shape == (7, 10)
    assert m[:, :4].all()  # prefix + root
    assert m[0, 4:].tolist() == [False] * 6
    assert m[5, 4:].tolist() == a[4].tolist()
    d = draft_mask(5, t.parents, rows=[2, 3], n_slots=8, device="cpu")
    assert d.shape == (2, 13) and d[:, :5].all()
    assert d[0, 5:11].tolist() == a[2].tolist() and not d[0, 11:].any()


def test_longest_accepted():
    t = _tree()
    # target says: root->10, n0->20, n2->31, n5->99  => path n0, n2, n5 ; bonus 99
    argmax = [10, 20, 0, 31, 0, 0, 99]
    assert longest_accepted(t, argmax) == ([0, 2, 5], 99)
    # root picks 11, n1 picks something not offered => path n1, bonus 77
    argmax = [11, 0, 77, 0, 0, 0, 0]
    assert longest_accepted(t, argmax) == ([1], 77)
    # root rejects everything
    assert longest_accepted(t, [5, 0, 0, 0, 0, 0, 0]) == ([], 5)
    assert t.ancestors(5) == [0, 2, 5]
