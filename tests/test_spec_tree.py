"""Static tree (topk=4, depth=5, 20 draft tokens). Output == the no-spec output
(tie-aware), accepted length >= the chain's, pool drained, and after every verify the
request's KV prefix equals what a plain decode writes for the same tokens (compaction)."""

from __future__ import annotations

import pytest
import torch

from engine.engine import Engine, EngineConfig
from kv.cache import AttnMeta
from sched.scheduler import SamplingParams
from tests.conftest import BF16_ULP, MAX_NEW_TOKENS, assert_tokens_match
from tests.test_batching import BLOCK_SIZE, NUM_BLOCKS
from tests.test_spec_chain import DEPTH, drafter, plain, prompts  # noqa: F401  (fixtures)

TOPK = 4


def _run(ns, drafter, prompts, topk, max_admit):
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=max_admit, cuda_graphs=False,
                                  spec_depth=DEPTH, spec_topk=topk), drafter)
    reqs = [eng.add(p, SamplingParams(MAX_NEW_TOKENS)) for p in prompts]
    while eng.sched.has_work:
        eng.step()
        eng.alloc.check()
    assert eng.alloc.num_free == eng.usable_blocks
    steps = sum(len(r.accepted) for r in reqs)
    return reqs, sum(sum(r.accepted) for r in reqs) / max(steps, 1)


@pytest.mark.parametrize("max_admit", [1, 32], ids=lambda m: f"admit{m}")
def test_tree_is_lossless_and_beats_chain(ns, drafter, prompts, plain, max_admit):
    chain_reqs, chain_acc = _run(ns, drafter, prompts, 1, max_admit)
    tree_reqs, tree_acc = _run(ns, drafter, prompts, TOPK, max_admit)
    ties = [(i, *t) for i, r in enumerate(tree_reqs)
            if (t := assert_tokens_match(r.out_tokens, plain[i][0], plain[i][1], False, f"prompt {i} tree vs plain"))]
    print(f"\n[spec tree admit={max_admit}] accepted/step chain {chain_acc:.2f} tree {tree_acc:.2f}; ties {ties}")
    assert tree_acc >= chain_acc


def test_kv_compaction_matches_plain_decode(ns, drafter, prompts):
    """Prompt 0 through the tree engine; after each step, K/V over [0, committed) must
    equal a teacher-forced prefill of the same tokens (up to kernel noise)."""
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=1, cuda_graphs=False,
                                  spec_depth=DEPTH, spec_topk=TOPK), drafter)
    ref = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, cuda_graphs=False))
    req = eng.add(prompts[0], SamplingParams(max_tokens=48))
    checked = 0
    while eng.sched.has_work:
        eng.step()
        if req.state != "running" or not req.accepted or req.accepted[-1] == 0:
            continue
        seq = req.prompt_ids + req.out_tokens[:-1]  # tokens whose K/V are committed
        assert len(seq) == req.committed
        r = ref.add(seq, SamplingParams(max_tokens=2))
        ref.step()  # prefill only; r stays running so its table is readable
        for layer in (0, ns.config.num_hidden_layers - 1):
            blocks = lambda e, rq: torch.tensor(rq.table.blocks, device=e.device)
            k_ref, v_ref = ref.kv.gather(layer, blocks(ref, r), len(seq))
            k_ours, v_ours = eng.kv.gather(layer, blocks(eng, req), len(seq))
            for a, b, name in ((k_ours, k_ref, "K"), (v_ours, v_ref, "V")):
                tol = 8 * BF16_ULP * b.float().abs().amax()
                worst = (a.float() - b.float()).abs().max().item()
                assert worst <= tol, f"step {len(req.accepted)} layer {layer} {name}: max diff {worst:.4f} > {tol:.4f}"
        ref.sched.finish(r)
        checked += 1
    assert checked > 0
