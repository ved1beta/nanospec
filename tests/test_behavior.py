"""Behavior log-probs of lossy speculative sampling (spec/sampling.py::behavior_logprob).

1. Pure functions, a random Markov target on a 4-token vocab, chain and tree proposals
   from a perturbed draft. Under the deterministic relaxed rule (tau / top-k) mu is exact
   on the sampler's support but the support is incomplete: reweighting by
   prod_t p(a_t | s_t) / mu(a_t | s_t) recovers the target exactly on the sequences the
   sampler can emit and recovers total mass equal to the target's mass on that support,
   which is < 1. Under the power rule (accept with probability min(1, p / rest) ** alpha)
   the support is complete and reweighting recovers the whole target distribution.
2. The engine, T = 1: the per-position histogram reweighted by the cumulative importance
   weight matches plain sampling within seed noise for the power rule; for the relaxed
   rule the reweighted histogram is right up to the missing mass, and every per-token
   weight is <= 1 (mu = 1 on accepted argmax drafts).
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest
import torch

from engine.engine import Engine, EngineConfig
from sched.scheduler import SamplingParams
from spec.sampling import behavior_logprob, draw, process, walk
from spec.tree import Tree
from tests.test_batching import BLOCK_SIZE, NUM_BLOCKS
from tests.test_sampling import _hist, _kl
from tests.test_spec_chain import DEPTH, drafter, plain, prompts  # noqa: F401
from tests.test_spec_tree import TOPK

# ---------------------------------------------------------------------------- 1. pure functions


def _markov(V: int, seed: int):
    """Random target p(. | prefix) and a perturbed draft d(. | prefix), memoised per prefix."""
    g = torch.Generator().manual_seed(seed)
    P, D = {}, {}

    def p_of(prefix):
        if prefix not in P:
            P[prefix] = torch.softmax(torch.randn(V, generator=g) * 2, -1)
        return P[prefix]

    def d_of(prefix):
        if prefix not in D:
            D[prefix] = torch.softmax(p_of(prefix).log() + torch.randn(V, generator=g) * 1.5, -1)
        return D[prefix]

    return p_of, d_of


def _propose(kind: str, state: tuple, d_of, depth: int) -> Tree:
    if kind == "chain":
        toks, s = [], state
        for _ in range(depth):
            toks.append(int(d_of(s).argmax()))
            s = s + (toks[-1],)
        return Tree.chain(toks)
    top = d_of(state).topk(2).indices.tolist()  # two root children, one grandchild each
    kids = [int(d_of(state + (c,)).argmax()) for c in top]
    return Tree(top + kids, [-1, -1, 0, 1], [1, 1, 2, 2])


def _sample(kind: str, p_of, d_of, L: int, depth: int, params: SamplingParams, g: torch.Generator):
    """One sequence of L tokens from the speculative sampler with argmax proposals; the
    tokens, their target log-probs and their behavior log-probs, as the engine computes them."""
    exact, relaxed = params.acceptance == "exact", (params.tau, params.accept_topk) if params.acceptance == "relaxed" else None
    state, lps, mus = (), [], []
    while len(state) < L:
        t = _propose(kind, state, d_of, depth)
        ch = t.children()
        rows = [state] + [state + tuple(t.tokens[a] for a in t.ancestors(i)) for i in range(t.n)]
        pn = [float(p_of(rows[1 + t.parents[i]])[t.tokens[i]]) for i in range(t.n)]
        rank = [int((p_of(rows[1 + t.parents[i]]) > pn[i]).sum()) for i in range(t.n)]
        u = torch.rand(t.n + 1, generator=g).tolist()
        path, _, acc = walk(ch, pn, [1.0] * t.n, u[: t.n], rank, relaxed, params.alpha)
        last = path[-1] if path else -1
        final = "resample" if ch.get(last) else "bonus"
        p_row = p_of(rows[1 + last])
        q = torch.zeros(1, len(p_row))
        for c in ch.get(last, []):
            q[0, t.tokens[c]] = 1.0
        a = int(draw(p_row.log()[None], q, torch.tensor([u[-1]])))
        before = lambda i: [(acc[c], pn[c]) for c in ch[t.parents[i]][: ch[t.parents[i]].index(i)]]
        emitted = [(t.tokens[i], math.log(pn[i]), behavior_logprob("draft", math.log(pn[i]), before(i), acc[i], exact)) for i in path]
        lp = math.log(float(p_row[a]))
        emitted.append((a, lp, behavior_logprob(final, lp, [(acc[c], pn[c]) for c in ch.get(last, [])], 0.0, exact)))
        for tok, lp, mu in emitted[: L - len(state)]:
            state, lps, mus = state + (tok,), lps + [lp], mus + [mu]
    return state, lps, mus


def _estimate(kind: str, params: SamplingParams, V: int = 4, L: int = 3, n: int = 30000, seed: int = 1):
    """-> (exact sequence probs, raw frequencies, reweighted frequencies, mean w^2)."""
    p_of, d_of = _markov(V, seed=0)
    g = torch.Generator().manual_seed(seed)
    seqs = list(itertools.product(range(V), repeat=L))
    exact = torch.tensor([math.prod(float(p_of(x[:t])[x[t]]) for t in range(L)) for x in seqs])
    idx = {x: i for i, x in enumerate(seqs)}
    raw, wtd, w2 = torch.zeros(len(seqs)), torch.zeros(len(seqs)), 0.0
    for _ in range(n):
        x, lps, mus = _sample(kind, p_of, d_of, L, 2, params, g)
        w = math.exp(sum(lps) - sum(mus))
        raw[idx[x]] += 1
        wtd[idx[x]] += w
        w2 += w * w
    return exact, raw / n, wtd / n, w2 / n


def _tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float((a - b).abs().sum())


@pytest.mark.parametrize("kind", ["chain", "tree"])
@pytest.mark.parametrize("params", [SamplingParams(acceptance="relaxed", tau=0.3), SamplingParams(acceptance="relaxed", tau=0.05, accept_topk=2)],
                         ids=["tau0.3", "tau0.05-top2"])
def test_relaxed_is_exact_on_its_support_only(kind, params):
    exact, raw, wtd, _ = _estimate(kind, params)
    seen = raw > 0
    support = float(exact[seen].sum())
    tv_raw = _tv(raw, exact)
    on_support = float((wtd[seen] - exact[seen]).abs().max())
    print(f"\n[behavior {kind} relaxed tau={params.tau} top-k={params.accept_topk}] TV raw {tv_raw:.3f}; "
          f"support holds {support:.3f} of the target mass, reweighted mass {float(wtd.sum()):.3f}; worst on-support |err| {on_support:.4f}")
    assert tv_raw > 0.03  # the relaxed sampler is visibly biased
    assert support < 0.9  # ... and it cannot reach a real share of the target's sequences
    assert on_support < 0.01 and abs(float(wtd.sum()) - support) < 0.02  # exact where it has support


@pytest.mark.parametrize("kind", ["chain", "tree"])
@pytest.mark.parametrize("alpha", [0.5, 0.2])
def test_power_reweighting_recovers_target(kind, alpha):
    """Reweighted TV to the target within the Monte Carlo floor: the exact sampler's own
    TV at the same n, scaled by the weights' root second moment (both measured)."""
    n = 50000
    exact, raw0, _, _ = _estimate(kind, SamplingParams(acceptance="exact"), n=n, seed=2)
    exact, raw, wtd, w2 = _estimate(kind, SamplingParams(acceptance="power", alpha=alpha), n=n)
    floor = _tv(raw0, exact)
    tv_raw, tv_wtd = _tv(raw, exact), _tv(wtd, exact)
    print(f"\n[behavior {kind} power alpha={alpha}] TV raw {tv_raw:.3f} -> reweighted {tv_wtd:.4f} (MC floor {floor:.4f} x sqrt(E w^2) {math.sqrt(w2):.2f}); "
          f"mass {float(wtd.sum()):.3f}")
    assert tv_raw > 0.03
    assert tv_wtd <= 2 * floor * math.sqrt(w2) and abs(float(wtd.sum()) - 1) < 0.02


def test_exact_acceptance_has_unit_weights():
    exact, raw, wtd, w2 = _estimate("chain", SamplingParams(acceptance="exact"), n=5000)
    assert torch.equal(raw, wtd) and w2 == 1.0 and _tv(raw, exact) < 0.05


def test_behavior_logprob_cases():
    assert behavior_logprob("draft", -0.7, [], 0.4, True) == -0.7  # exact: mu = p everywhere
    assert behavior_logprob("resample", -0.7, [(0.0, 0.2)], 0.0, True) == -0.7
    assert behavior_logprob("draft", -0.7, [], 1.0, False) == 0.0  # relaxed: accepted draft is deterministic
    assert behavior_logprob("draft", -0.7, [(0.0, 0.2)], 1.0, False) == 0.0  # deterministic rejected sibling first
    assert behavior_logprob("bonus", -0.7, [], 0.0, False) == -0.7
    assert behavior_logprob("resample", math.log(0.4), [(0.0, 0.2), (0.0, 0.3)], 0.0, False) == pytest.approx(math.log(0.4 / 0.5))
    # power: accepted second sibling after the first was rejected with accept prob 0.6
    assert behavior_logprob("draft", -0.7, [(0.6, 0.2)], 0.5, False) == pytest.approx(math.log(0.4 * 0.5))
    assert behavior_logprob("resample", math.log(0.4), [(0.6, 0.2)], 0.0, False) == pytest.approx(math.log(0.4 * 0.4 / 0.8))


# ---------------------------------------------------------------------------- 2. the engine


def _hists(reqs, positions: int, top: torch.Tensor):
    """Per position: the raw histogram over `top` (+other), the same weighted by the
    cumulative importance weight, the sum of squared weights, and the largest per-token weight."""
    raw, wtd, w2, wmax = torch.zeros(positions, len(top) + 1), torch.zeros(positions, len(top) + 1), torch.zeros(positions, len(top) + 1), 0.0
    for r in reqs:
        w = 1.0
        for k in range(min(positions, len(r.out_tokens))):  # eos can end a sequence early
            info = r.logprobs[k]
            wt = math.exp(info.sample_logprob - info.behavior_logprob)
            w, wmax = w * wt, max(wmax, wt)
            j = (top == r.out_tokens[k]).nonzero().flatten()
            raw[k, int(j) if len(j) else -1] += 1
            wtd[k, int(j) if len(j) else -1] += w
            w2[k, int(j) if len(j) else -1] += w * w
    return raw, wtd, w2, wmax


@pytest.mark.parametrize("params", [SamplingParams(acceptance="power", alpha=0.3), SamplingParams(acceptance="relaxed", tau=0.3, accept_topk=2)],
                         ids=["power0.3", "relaxed-tau0.3-top2"])
@pytest.mark.parametrize("topk", [1, TOPK], ids=["chain", "tree"])
def test_engine_reweighting(ns, drafter, prompts, params, topk):
    from kv.cache import AttnMeta

    n, positions = 1024, 3
    params = dataclasses.replace(params, max_tokens=positions, temperature=1.0, logprobs=True, seed=0)
    cfg = lambda depth, k: EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=n, cuda_graphs=False, spec_depth=depth, spec_topk=k)
    plain_a = Engine(ns, cfg(0, 1), None)
    with torch.no_grad():
        t = plain_a.alloc.alloc(-5, len(prompts[3]))
        lg, _ = ns(torch.tensor(prompts[3], device=plain_a.device), plain_a.kv, AttnMeta.build([t], [len(prompts[3])], BLOCK_SIZE, plain_a.device))
        plain_a.alloc.free(-5)
    z = lg[-1:].float()
    one = lambda v: torch.tensor([v], device=z.device)
    p0 = process(z, one(1.0), one(0), one(1.0))[0].exp().cpu()
    top = z[0].topk(30).indices.cpu()
    ha = _hist(plain_a, prompts[3], dataclasses.replace(params, acceptance="exact", seed=1), n, positions, top)
    hb = _hist(Engine(ns, cfg(0, 1), None), prompts[3], dataclasses.replace(params, acceptance="exact", seed=2), n, positions, top)
    spec = Engine(ns, cfg(DEPTH, topk), drafter)
    reqs = [spec.add(prompts[3], dataclasses.replace(params, seed=3000 + i)) for i in range(n)]
    while spec.sched.has_work:
        spec.step()
    hs, hw, hw2, wmax = _hists(reqs, positions, top)
    ref = (hb + 0.5) / (hb + 0.5).sum(-1, keepdim=True)
    ref[0] = torch.cat([p0[top], 1 - p0[top].sum()[None]]).clamp_min(0.5 / n)
    noise = max(_kl(ha[k], ref[k]) for k in range(positions))
    biased = max(_kl(hs[k], ref[k]) for k in range(positions))
    # a weighted histogram of n samples has the noise of (sum w)^2 / sum w^2 unweighted ones
    fixed = max(_kl(hw[k], ref[k]) * min(1.0, float(hw[k].sum()) ** 2 / (float(hw2[k].sum()) * n)) for k in range(positions))
    mass = [float(hw[k].sum()) / n for k in range(positions)]
    acc = sum(sum(r.accepted) for r in reqs) / max(sum(len(r.accepted) for r in reqs), 1)
    print(f"\n[behavior engine {'tree' if topk > 1 else 'chain'} {params.acceptance}] KL biased {biased:.4f} -> reweighted {fixed:.4f} (ESS-scaled) "
          f"vs plain noise {noise:.4f}; reweighted mass per position {[f'{m:.3f}' for m in mass]}; max w {wmax:.3f}; accepted/step {acc:.2f}")
    assert all(info.behavior_logprob <= 1e-9 for r in reqs for info in r.logprobs)
    if params.acceptance == "relaxed":
        assert wmax <= 1 + 1e-6  # mu = 1 on accepted drafts, so every weight is a probability
        # exact on its support: position 0's reweighted counts equal p0 on the tokens the sampler emits
        seen = hs[0] > 0
        err = ((hw[0] / n)[seen] - ref[0][seen]).abs().max()
        assert err < 0.02 and mass[0] <= 1.02
    else:
        assert fixed <= 2 * noise + 0.01 and all(abs(m - 1) < 0.05 for m in mass)
