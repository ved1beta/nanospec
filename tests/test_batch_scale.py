"""Speculation at 64 prompts x 8 samples = 512 concurrent sequences, continuously batched:
admission is capped by max_running and by the KV reservation (prompt + max_tokens +
draft slots), so append() never fails mid-decode and the pool drains at the end. The
per-step chain depth (spec_row_budget) stays lossless at T=0."""

from __future__ import annotations

import pytest
import torch

from engine.engine import Engine, EngineConfig
from kv.allocator import PoolExhausted
from sched.scheduler import SamplingParams
from tests.conftest import assert_tokens_match
from tests.test_batching import BLOCK_SIZE
from tests.test_spec_chain import DEPTH, drafter, plain, prompts  # noqa: F401


@pytest.mark.parametrize("topk", [1, 4], ids=["chain", "tree"])
def test_64x8_concurrent(ns, drafter, prompts, topk):
    G, S, max_tokens = 64, 8, 16
    ps = [prompts[i % len(prompts)] for i in range(G)]
    need = sum(-(-(len(p) + max_tokens + DEPTH * topk) // BLOCK_SIZE) for p in ps) * S
    eng = Engine(ns, EngineConfig(need + 8, BLOCK_SIZE, max_admit=128, max_running=G * S, cuda_graphs=False,
                                  spec_depth=DEPTH, spec_topk=topk), drafter)
    reqs = [eng.add(p, SamplingParams(max_tokens, temperature=1.0, seed=g * S + s, logprobs=True)) for g, p in enumerate(ps) for s in range(S)]
    steps, peak = 0, 0
    while eng.sched.has_work:
        eng.step()
        steps += 1
        peak = max(peak, len(eng.sched.running))
        assert len(eng.sched.running) <= G * S
    eng.alloc.check()
    assert eng.alloc.num_free == eng.usable_blocks
    assert all(r.state == "done" and 0 < len(r.out_tokens) <= max_tokens and len(r.logprobs) == len(r.out_tokens) for r in reqs)
    acc = sum(sum(r.accepted) for r in reqs) / sum(len(r.accepted) for r in reqs)
    print(f"\n[64x8 {'tree' if topk > 1 else 'chain'}] {steps} steps, peak {peak} running, {acc:.2f} accepted/step")
    assert peak == G * S  # the pool was sized for all of them at once


def test_small_pool_waits_instead_of_failing(ns, drafter, prompts):
    """A pool for ~6 max-length requests: admission throttles, nothing raises."""
    ps = prompts[:24]
    blocks = 6 * -(-(max(map(len, ps)) + 32 + DEPTH) // BLOCK_SIZE)
    eng = Engine(ns, EngineConfig(blocks, BLOCK_SIZE, max_admit=8, cuda_graphs=False, spec_depth=DEPTH), drafter)
    reqs = [eng.add(p, SamplingParams(32, temperature=0.7, seed=i)) for i, p in enumerate(ps)]
    peak = 0
    try:
        while eng.sched.has_work:
            eng.step()
            eng.alloc.check()
            peak = max(peak, len(eng.sched.running))
    except PoolExhausted as e:
        pytest.fail(f"append failed mid-decode: {e}")
    assert all(r.state == "done" for r in reqs) and eng.alloc.num_free == eng.usable_blocks
    smallest = -(-(min(map(len, ps)) + 32 + DEPTH) // BLOCK_SIZE)
    assert 1 <= peak <= blocks // smallest < len(ps), (peak, blocks // smallest)


def test_row_budget_is_lossless(ns, drafter, prompts, plain):
    """spec_row_budget=64: depth 2 at batch 32, back to 5 as the batch drains; greedy-exact throughout."""
    from tests.test_batching import NUM_BLOCKS
    from tests.conftest import MAX_NEW_TOKENS

    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=32, cuda_graphs=False, spec_depth=DEPTH, spec_row_budget=64), drafter)
    reqs = [eng.add(p, SamplingParams(MAX_NEW_TOKENS)) for p in prompts]
    depths = set()
    while eng.sched.has_work:
        eng.step()
        depths |= {r.tree.n for r in eng.sched.running if r.tree is not None}
    for i, r in enumerate(reqs):
        assert_tokens_match(r.out_tokens, plain[i][0], plain[i][1], False, f"prompt {i} budgeted chain vs plain")
    print(f"\n[row budget] chain depths seen {sorted(depths)}")
    assert len(depths) > 1 and min(depths) < DEPTH
