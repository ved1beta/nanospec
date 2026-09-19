"""Stochastic acceptance, per-token log-probs, telemetry, weight updates.

1. Pure functions, tiny vocab, Monte Carlo: chain / tree / sampled-proposal acceptance
   emit exactly the target distribution under adversarial proposals (KL < 1e-3).
2. The engine, T > 0: the token histogram at each of the first positions from the
   speculative engine matches plain sampling within 3x the plain-vs-plain seed noise.
3. Greedy log-probs equal the reference log-softmax; draft log-probs equal the target's
   for the self-drafter (whose proposals are the target's own logits).
4. Telemetry lines are consistent with Request.accepted.
5. update_weights / swap_drafter mid-generation: identical to an uninterrupted run with
   the new weights (weights copied in place, running requests re-prefilled)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from engine.engine import Engine, EngineConfig
from sched.scheduler import SamplingParams
from spec.sampling import draw, process, walk
from spec.tree import Tree
from tests.conftest import MAX_NEW_TOKENS, assert_tokens_match, logit_tol
from tests.test_batching import BLOCK_SIZE, NUM_BLOCKS
from tests.test_spec_chain import DEPTH, EAGLE, SelfDrafter, drafter, plain, prompts  # noqa: F401
from tests.test_spec_tree import TOPK

# ---------------------------------------------------------------------------- 1. pure functions


def _kl(counts: torch.Tensor, p: torch.Tensor) -> float:
    e = (counts + 0.5) / (counts + 0.5).sum()
    return float((e * (e / p).log()).sum())


def _emit_first(p_rows, tree: Tree, q_rows=None, n=40000):
    """Monte Carlo of one verify step with p_rows[r] the target distribution at row r
    (row 0 = root, row 1 + i = node i) and q_rows[r] the proposal distribution at row r
    (None = point masses): histogram of the first emitted token."""
    V, g = p_rows.shape[-1], torch.Generator().manual_seed(0)
    par = torch.tensor(tree.parents)
    nrow = 1 + par
    p_node = p_rows[nrow, tree.tokens]
    q_node = q_rows[nrow, tree.tokens] if q_rows is not None else torch.ones(tree.n)
    q = torch.zeros(len(p_rows), V)
    if q_rows is None:
        q[nrow, tree.tokens] = 1.0
    else:
        q = q_rows.clone()
    counts = torch.zeros(V)
    for _ in range(n):
        u = torch.rand(tree.n + len(p_rows), generator=g)
        path, _, _ = walk(tree.children(), p_node.tolist(), q_node.tolist(), u[: tree.n].tolist(), None, None)
        if path:
            counts[tree.tokens[path[0]]] += 1
        else:
            counts[draw(p_rows[:1].log(), q[:1], u[tree.n : tree.n + 1])] += 1
    return counts


def test_point_mass_chain_and_tree_are_exact():
    torch.manual_seed(0)
    V = 6
    p = torch.softmax(torch.randn(8, V) * 2, -1)
    # chain of depth 3 whose drafts are the target's LEAST likely tokens (adversarial)
    chain = Tree.chain(p[:3].argmin(-1).tolist())
    assert _kl(_emit_first(p, chain), p[0]) < 1e-3
    # tree: root has 3 children (tokens 0, 1, 2), node 0 has two more
    tree = Tree([0, 1, 2, 3, 4], [-1, -1, -1, 0, 0], [1, 1, 1, 2, 2])
    assert _kl(_emit_first(p, tree), p[0]) < 1e-3
    # sibling order matters for the residual: the tree's most likely child first
    p[0] = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    assert _kl(_emit_first(p, tree), p[0]) < 1e-3


def test_sampled_proposal_is_exact():
    """Leviathan: proposals drawn from q != p, accept u < min(1, p/q), residual (p-q)+."""
    torch.manual_seed(1)
    V, g = 6, torch.Generator().manual_seed(2)
    p = torch.softmax(torch.randn(2, V) * 2, -1)
    q = torch.softmax(torch.randn(2, V) * 2, -1)
    counts = torch.zeros(V)
    for _ in range(40000):
        d = int(torch.multinomial(q[0], 1, generator=g))
        u = torch.rand(2, generator=g)
        path, _, _ = walk({-1: [0]}, [float(p[0, d])], [float(q[0, d])], [float(u[0])], None, None)
        counts[d if path else int(draw(p[:1].log(), q[:1], u[1:2]))] += 1
    assert _kl(counts, p[0]) < 1e-3


def test_process_matches_reference():
    z = torch.randn(4, 50)
    T, k, pp = torch.tensor([1.0, 0.7, 1.0, 0.0]), torch.tensor([0, 5, 0, 0]), torch.tensor([1.0, 1.0, 0.8, 1.0])
    lp = process(z, T, k, pp)
    torch.testing.assert_close(lp[0], z[0].log_softmax(-1))
    torch.testing.assert_close(lp[3], z[3].log_softmax(-1))  # greedy rows: scaled by 1
    top5 = (z[1] / 0.7).topk(5)
    assert (lp[1] > -math.inf).sum() == 5 and torch.allclose(lp[1][top5.indices], top5.values.log_softmax(-1))
    srt = z[2].softmax(-1).sort(descending=True)
    keep = (srt.values.cumsum(-1) - srt.values) <= 0.8
    assert set((lp[2] > -math.inf).nonzero().flatten().tolist()) == set(srt.indices[keep].tolist())


# ---------------------------------------------------------------------------- 2. the engine


def _hist(eng: Engine, prompt, params: SamplingParams, n: int, positions: int, top: torch.Tensor):
    reqs = [eng.add(prompt, dataclasses.replace(params, seed=1000 * params.seed + i)) for i in range(n)]
    while eng.sched.has_work:
        eng.step()
    counts = torch.zeros(positions, len(top) + 1)
    for r in reqs:
        for k in range(positions):
            j = (top == r.out_tokens[k]).nonzero().flatten()
            counts[k, int(j) if len(j) else -1] += 1
    return counts


class FlatDrafter(SelfDrafter):
    """Proposals from a flattened target: q far from p, so a wrong residual shows."""

    def __call__(self, ids, hidden, kv, meta, **kw):
        logits, h = super().__call__(ids, hidden, kv, meta, **kw)
        return logits / 3, h


@pytest.mark.parametrize("params", [
    SamplingParams(max_tokens=3, temperature=1.0),
    SamplingParams(max_tokens=3, temperature=0.8, top_p=0.9),
    SamplingParams(max_tokens=3, temperature=1.0, draft_sampling="sample"),
], ids=["T1", "T0.8-p0.9", "sampled-drafts"])
@pytest.mark.parametrize("topk", [1, TOPK], ids=["chain", "tree"])
def test_spec_sampling_matches_plain(ns, drafter, prompts, params, topk):
    """n samples of one prompt; per position, the histogram over the target's top-30
    next tokens (+other). Position 0 is judged against the exact processed distribution,
    later positions (the speculated ones) against an independent plain run; the
    speculative engine's KL must be within 2x the plain engine's own noise (+0.01)."""
    if topk > 1 and params.draft_sampling == "sample":
        pytest.skip("sampled proposals are chain-only")
    if params.draft_sampling == "sample" and not EAGLE:
        drafter = FlatDrafter(ns)
    n, positions = 1024, 3
    cfg = lambda depth, k: EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=n, cuda_graphs=False, spec_depth=depth, spec_topk=k)
    plain_a = Engine(ns, cfg(0, 1), None)
    from kv.cache import AttnMeta  # the target's top-30 next tokens at position 0 define the bins

    with torch.no_grad():
        t = plain_a.alloc.alloc(-5, len(prompts[3]))
        lg, _ = ns(torch.tensor(prompts[3], device=plain_a.device), plain_a.kv, AttnMeta.build([t], [len(prompts[3])], BLOCK_SIZE, plain_a.device))
        plain_a.alloc.free(-5)
    z = lg[-1:].float()
    p0 = process(z, torch.tensor([params.temperature], device=z.device), torch.tensor([params.top_k], device=z.device),
                 torch.tensor([params.top_p], device=z.device))[0].exp().cpu()
    top = z[0].topk(30).indices.cpu()
    ha = _hist(plain_a, prompts[3], dataclasses.replace(params, seed=1), n, positions, top)
    hb = _hist(Engine(ns, cfg(0, 1), None), prompts[3], dataclasses.replace(params, seed=2), n, positions, top)
    hs = _hist(Engine(ns, cfg(DEPTH, topk), drafter), prompts[3], dataclasses.replace(params, seed=3), n, positions, top)
    ref = (hb + 0.5) / (hb + 0.5).sum(-1, keepdim=True)
    ref[0] = torch.cat([p0[top], 1 - p0[top].sum()[None]]).clamp_min(0.5 / n)  # same floor as the +0.5 smoothing
    noise = max(_kl(ha[k], ref[k]) for k in range(positions))
    worst = max(_kl(hs[k], ref[k]) for k in range(positions))
    print(f"\n[sampling {'tree' if topk > 1 else 'chain'} T={params.temperature} top_p={params.top_p} {params.draft_sampling}] "
          f"KL spec {worst:.4f} vs plain {noise:.4f}")
    assert worst <= 2 * noise + 0.01


def test_relaxed_accepts_more_and_seed_is_reproducible(ns, drafter, prompts):
    cfg = EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=8, cuda_graphs=False, spec_depth=DEPTH)
    exact = SamplingParams(max_tokens=32, temperature=1.0, seed=7)
    relaxed = SamplingParams(max_tokens=32, temperature=1.0, seed=7, acceptance="relaxed", tau=0.05, accept_topk=3)
    rate = lambda reqs: sum(sum(r.accepted) for r in reqs) / sum(len(r.accepted) for r in reqs)

    def go(p):
        eng = Engine(ns, cfg, drafter)
        reqs = [eng.add(pr, p) for pr in prompts[:8]]
        while eng.sched.has_work:
            eng.step()
        return reqs

    a, b, c = go(exact), go(exact), go(relaxed)
    assert [r.out_tokens for r in a] == [r.out_tokens for r in b]  # same seeds, same tokens
    assert rate(c) >= rate(a), (rate(c), rate(a))
    one = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=1, cuda_graphs=False, spec_depth=DEPTH), drafter)
    solo = [one.add(pr, exact) for pr in prompts[:8]]
    while one.sched.has_work:
        one.step()
    same = sum(r.out_tokens == s.out_tokens for r, s in zip(a, solo))
    print(f"\n[sampling] exact accepted/step {rate(a):.2f}, relaxed {rate(c):.2f}; {same}/8 identical across batch composition")


# ---------------------------------------------------------------------------- 3. log-probs


@pytest.mark.parametrize("topk", [0, 1, TOPK], ids=["plain", "chain", "tree"])
def test_greedy_logprobs_match_reference(ns, drafter, prompts, plain, topk):
    """Emitted log-probs equal the reference log-softmax at every step (up to a bf16 tie,
    the engine is batched); accepted drafts' q equal a teacher-forced drafter forward
    over the request's own sequence (the self-drafter reads it from token 1 on)."""
    from kv.cache import AttnMeta

    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, cuda_graphs=False, spec_depth=DEPTH if topk else 0,
                                  spec_topk=max(topk, 1)), drafter if topk else None)
    reqs = [eng.add(p, SamplingParams(MAX_NEW_TOKENS, logprobs=True)) for p in prompts]
    while eng.sched.has_work:
        eng.step()
    worst, worst_q, n_draft = 0.0, 0.0, 0
    for i, (r, prompt) in enumerate(zip(reqs, prompts)):
        ref_tokens, ref_logits = plain[i]
        tie = assert_tokens_match(r.out_tokens, ref_tokens, ref_logits, False, f"prompt {i}")
        n = min(len(r.logprobs), tie[0] if tie else len(ref_logits))
        assert len(r.logprobs) == len(r.out_tokens)
        lq = None
        if topk and not EAGLE:
            ds = prompt[1:] + r.out_tokens
            t = eng.alloc.alloc(-9, len(ds))
            meta = AttnMeta.build([t], [len(ds)], BLOCK_SIZE, eng.device)
            lq = drafter(torch.tensor(ds, device=eng.device), None, eng.draft_kv, meta)[0].float().log_softmax(-1).cpu()
            eng.alloc.free(-9)
        for k in range(n):
            info, row = r.logprobs[k], ref_logits[k].float()
            assert info.token == r.out_tokens[k] and info.sample_logprob == 0.0
            ref = row.log_softmax(-1)[info.token].item()
            tol = 2 * logit_tol(row).item()
            assert abs(info.logprob - ref) <= tol, f"prompt {i} step {k} ({info.source}): {info.logprob:.4f} vs {ref:.4f}"
            worst = max(worst, abs(info.logprob - ref))
            if info.source == "draft":
                n_draft += 1
                if lq is not None:  # drafter row m-1 predicts ds[m]
                    m = len(prompt) - 1 + k
                    dq = abs(info.draft_logprob - lq[m - 1, info.token].item())
                    assert dq <= 2 * logit_tol(lq[m - 1]).item(), f"prompt {i} step {k}: draft q {info.draft_logprob:.4f} vs {lq[m - 1, info.token]:.4f}"
                    worst_q = max(worst_q, dq)
            else:
                assert info.draft_logprob is None and info.source in ("bonus", "resample")
    print(f"\n[logprobs {['plain', 'chain', 'tree'][min(topk, 2)]}] worst |logprob - ref| {worst:.5f}, "
          f"{n_draft} draft tokens, worst |q - teacher-forced| {worst_q:.5f}")
    assert topk == 0 or n_draft > 0


# ---------------------------------------------------------------------------- 4. telemetry


def test_telemetry_lines(ns, drafter, prompts, tmp_path):
    path = tmp_path / "acc.jsonl"
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=4, cuda_graphs=False, spec_depth=DEPTH,
                                  spec_topk=TOPK, telemetry_path=str(path)), drafter)
    reqs = [eng.add(p, SamplingParams(24, temperature=0.5, seed=0)) for p in prompts[:6]]
    while eng.sched.has_work:
        eng.step()
    eng.tele.close()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    by_req = {}
    for row in rows:
        assert len(row["nodes"]) == DEPTH * TOPK and row["final"] in ("bonus", "resample")
        assert row["accepted"] == sum(nd["d"] == "accept" for nd in row["nodes"])
        assert all(0 <= nd["p"] <= 1 and nd["q"] <= 0 and 0 <= nd["u"] < 1 for nd in row["nodes"])
        by_req.setdefault(row["req"], []).append(row["accepted"])
    assert {r.id: r.accepted for r in reqs} == by_req
    # verify steps start one step after admission; the second batch of 2 is admitted a step after the first 4
    longest = max(len(r.accepted) for r in reqs)
    assert len({row["step"] for row in rows}) in (longest, longest + 1)


# ---------------------------------------------------------------------------- 5. weights


def test_update_weights_mid_generation(ns, drafter, prompts, plain):
    """Restart with the same weights is lossless; after an update the continuation is
    what a fresh engine on the new weights produces from the same prefix (tie-aware: the
    engine batches 8, the references run one at a time). The update is one layer
    perturbed in place (no model copies: the 8B does not fit twice) and restored."""
    cfg = EngineConfig(NUM_BLOCKS, BLOCK_SIZE, cuda_graphs=False, spec_depth=DEPTH)
    eng = Engine(ns, cfg, drafter)
    reqs = [eng.add(p, SamplingParams(MAX_NEW_TOKENS, logprobs=True)) for p in prompts[:8]]
    for _ in range(3):
        eng.step()
    eng.update_weights({})  # same weights: the 8 running requests restart, losslessly
    assert all(r.state == "waiting" and r.out_tokens for r in reqs)
    while eng.sched.has_work:
        eng.step()
        eng.alloc.check()
    for i, r in enumerate(reqs):
        assert len(r.logprobs) == len(r.out_tokens)
        assert_tokens_match(r.out_tokens, plain[i][0], plain[i][1], False, f"prompt {i} after same-weight restart")

    torch.manual_seed(0)
    layer = {k: v.clone() for k, v in ns.state_dict().items() if k.startswith("model.layers.1.")}
    bumped = {k: v * (1.0 + 0.05 * torch.randn_like(v).clamp(-2, 2)) for k, v in layer.items()}
    eng = Engine(ns, cfg, drafter)
    reqs = [eng.add(p, SamplingParams(40)) for p in prompts[:8]]
    for _ in range(3):
        eng.step()
    prefix = [list(r.out_tokens) for r in reqs]
    eng.update_weights(bumped)
    try:
        while eng.sched.has_work:
            eng.step()
        assert eng.alloc.num_free == eng.usable_blocks
        ref = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, cuda_graphs=False, record_logits=True))
        for i, (r, p, pre) in enumerate(zip(reqs, prompts[:8], prefix)):
            assert r.out_tokens[: len(pre)] == pre
            want = ref.add(p + pre, SamplingParams(40 - len(pre)))
            while ref.sched.has_work:
                ref.step()
            assert_tokens_match(r.out_tokens[len(pre) :], want.out_tokens, torch.stack(want.step_logits), False, f"prompt {i} after update")
    finally:
        eng.update_weights(layer)  # restore the session model bit-exactly


def test_swap_drafter(ns, drafter, prompts, plain):
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=8, cuda_graphs=False, spec_depth=DEPTH, spec_topk=TOPK), drafter)
    reqs = [eng.add(p, SamplingParams(MAX_NEW_TOKENS)) for p in prompts[:8]]
    for _ in range(4):
        eng.step()
    eng.swap_drafter(SelfDrafter(ns))  # a different object: rebind + restart
    assert all(r.state == "waiting" for r in reqs) and eng.alloc.num_free == eng.usable_blocks
    while eng.sched.has_work:
        eng.step()
    for i, r in enumerate(reqs):
        assert_tokens_match(r.out_tokens, plain[i][0], plain[i][1], False, f"prompt {i} after swap")
