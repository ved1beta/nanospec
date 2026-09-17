"""Speculative step with the verify forward captured. Same outputs as the eager spec
engine (tie-aware); bs=1 tok/s with tree + graphs >= 2x the no-spec + graphs number."""

from __future__ import annotations

import time

import pytest
import torch

from engine.engine import Engine, EngineConfig
from sched.scheduler import SamplingParams
from tests.conftest import DEVICE, MAX_NEW_TOKENS, assert_tokens_match
from tests.test_batching import BLOCK_SIZE, NUM_BLOCKS
from tests.test_spec_chain import DEPTH, drafter, plain, prompts  # noqa: F401
from tests.test_spec_tree import TOPK

pytestmark = pytest.mark.skipif(DEVICE.type != "cuda", reason="CUDA graphs need a GPU")


def _engine(ns, drafter, topk, graphs, max_admit=1, profile=False):
    return Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=max_admit, cuda_graphs=graphs,
                                   spec_depth=DEPTH if topk else 0, spec_topk=max(topk, 1), profile=profile),
                  drafter if topk else None)


@pytest.mark.parametrize("topk", [1, TOPK], ids=["chain", "tree"])
def test_graphs_match_eager_spec(ns, drafter, prompts, plain, topk):
    eager = _engine(ns, drafter, topk, False)
    graphed = _engine(ns, drafter, topk, True)
    a = eager.generate(prompts, SamplingParams(MAX_NEW_TOKENS))
    b = graphed.generate(prompts, SamplingParams(MAX_NEW_TOKENS))
    ties = [(i, *t) for i, (x, y) in enumerate(zip(b, a))
            if (t := assert_tokens_match(x, plain[i][0], plain[i][1], False, f"prompt {i} spec graphs vs plain"))]
    assert graphed.alloc.num_free == graphed.usable_blocks
    print(f"\n[spec graphs {'tree' if topk > 1 else 'chain'}] ties {ties}")


@pytest.mark.parametrize("topk", [1, TOPK], ids=["chain", "tree"])
def test_verify_graph_matches_eager_per_step(ns, drafter, prompts, topk):
    """Teacher-forced: at every step run the eager verify forward and the captured one on
    the identical meta/ids and compare logits + aux. Includes a prompt whose length equals
    the row count R (prefill through the graph) and staggered batches."""
    eng = _engine(ns, drafter, topk, True, max_admit=1)
    R = eng.config.num_draft + 1
    tok_len_R = next((p for p in prompts if len(p) == R), None)
    ps = ([tok_len_R] if tok_len_R else []) + prompts[:6]
    reqs = [eng.add(p, SamplingParams(max_tokens=24 + 4 * i)) for i, p in enumerate(ps)]
    worst, n_graph = 0.0, 0
    orig = eng._verify_forward

    def both(ids, meta):
        import torch as _t

        input_ids = _t.tensor(ids, dtype=_t.int64, device=eng.device)
        lg_e, aux_e = ns(input_ids, eng.kv, eng._materialize(meta), aux_layers=eng.taps)
        aux_e = _t.cat(aux_e, -1)
        if eng.verify_graphs.can_run(meta):
            nonlocal worst, n_graph
            lg_g, aux_g = eng.verify_graphs.run(meta, ids)
            u = ((lg_g.float() - lg_e.float()).abs() / (2.0**-7 * lg_e.float().abs().amax(-1, keepdim=True))).max().item()
            ua = ((aux_g.float() - aux_e.float()).abs() / (2.0**-7 * aux_e.float().abs().amax(-1, keepdim=True))).max().item()
            worst, n_graph = max(worst, u, ua), n_graph + 1
            # FA2 masked (graph) vs FA3 causal (eager) read 2-30 ulps here; the "no mask"
            # bug read 90-137. The lossless end-to-end tests above are the gate.
            assert u <= 64 and ua <= 64, f"rows {len(ids)} qo {meta.qo_lens} kv {meta.seq_lens}: logits {u:.2f} aux {ua:.2f} ulps"
        return lg_e, aux_e

    eng._verify_forward = both
    while eng.sched.has_work:
        eng.step()
    eng._verify_forward = orig
    print(f"\n[spec graphs {'tree' if topk > 1 else 'chain'} per-step] graph vs eager worst {worst:.2f} ulps over {n_graph} captured steps")
    assert n_graph > 0


def test_verify_runner_matrix(ns, prompts):
    """VerifyGraphRunner vs eager on hand-built metas: rows at position 0 (prefill through
    the graph) vs mid-sequence, causal-converted vs explicit masks, buckets 1 and 4, and
    both orders of first use."""
    from engine.graphs import GraphRunner
    from kv.cache import AttnMeta
    from spec.tree import verify_mask

    R = DEPTH + 1
    eng = _engine(ns, None, 0, False)  # just for kv/alloc; no graphs
    pad = eng.alloc.alloc(-1, 1).blocks[0]
    c = ns.config
    runner = GraphRunner(eng.kv, pad, R, True, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                         eng.dtype, eng.device, buckets=(1, 4))
    runner.taps = eng.taps
    runner.capture(lambda ids, _h, meta, be: (lambda lg, aux: (lg, torch.cat(aux, -1)))(*ns(ids, eng.kv, meta, aux_layers=eng.taps, backend=be)))
    torch.manual_seed(0)

    def case(name, B, start, explicit):
        reqs = []
        for b in range(B):
            L = start + 3 * b
            t = eng.alloc.alloc(100 + b, L + R)
            if L:  # fill a prefix so mid-sequence rows attend to real K/V
                m0 = AttnMeta.build([t], [L], BLOCK_SIZE, eng.device, [0])
                ns(torch.tensor(prompts[b][:1] * L, device=eng.device), eng.kv, m0)
            reqs.append((t, L))
        rows = [([*range(L, L + R)], [*range(L, L + R)], L + R) for _, L in reqs]
        masks = [verify_mask(L, list(range(-1, R - 2)), eng.device) if explicit else None for _, L in reqs]
        meta = AttnMeta.from_rows([t for t, _ in reqs], rows, BLOCK_SIZE, eng.device, masks)
        ids = [prompts[0][i % len(prompts[0])] for i in range(B * R)]
        lg_e, aux_e = ns(torch.tensor(ids, device=eng.device), eng.kv, meta, aux_layers=runner.taps)
        aux_e = torch.cat(aux_e, -1)
        lg_g, aux_g = runner.run(meta, ids)
        u = ((lg_g.float() - lg_e.float()).abs() / (2.0**-7 * lg_e.float().abs().amax(-1, keepdim=True))).max().item()
        ua = ((aux_g.float() - aux_e.float()).abs() / (2.0**-7 * aux_e.float().abs().amax(-1, keepdim=True))).max().item()
        for b in range(B):
            eng.alloc.free(100 + b)
        print(f"\n[spec graphs runner] {name:28s} B={B} start={start} explicit={explicit}: logits {u:6.2f} aux {ua:6.2f} ulps")
        return u

    results = {
        "mid first": case("mid-seq causal", 1, 40, False),
        "prefill after": case("prefill causal", 1, 0, False),
        "prefill explicit": case("prefill explicit mask", 1, 0, True),
        "mid explicit": case("mid explicit mask", 1, 40, True),
        "prefill b4": case("prefill causal", 4, 0, False),
        "mid b4": case("mid-seq causal", 4, 40, False),
        "prefill again": case("prefill causal", 1, 0, False),
    }
    bad = {k: v for k, v in results.items() if v > 8}  # FA2 (graph) vs FA3 (eager causal) kernel noise; bugs read 90+
    assert not bad, bad


def _tok_s(eng, prompt, max_tokens=200):
    r = eng.add(prompt, SamplingParams(max_tokens))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while eng.sched.has_work:
        eng.step()
    torch.cuda.synchronize()
    return len(r.out_tokens) / (time.perf_counter() - t0), r


def test_bs1_tok_s_2x_over_nospec(ns, drafter, hf):
    """On the benchmark workload (chat-templated MT-Bench), not on raw fragments."""
    from bench.prompts import chat_prompts

    tok, _ = hf
    prompt = tok(chat_prompts(tok)[0], add_special_tokens=False).input_ids
    base, _ = _tok_s(_engine(ns, None, 0, True), prompt)
    acc = lambda r: sum(r.accepted) / max(len(r.accepted), 1) + 1
    prof = lambda e: " ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in e.prof.items())
    best = 0.0
    print(f"\n[spec graphs] bs=1 tok/s: no-spec+graphs {base:.1f}")
    for topk in (1, TOPK, 6):  # 6 x 5 = 30 draft tokens, still within PLAN D5's <= 32
        e = _engine(ns, drafter, topk, True, profile=True)
        tps, r = _tok_s(e, prompt)
        best = max(best, tps)
        print(f"[spec graphs] topk={topk}: {tps:.1f} tok/s ({acc(r):.2f} tok/step, {tps / base:.2f}x)  ms/step: {prof(e)}")
    assert best >= 2.0 * base
