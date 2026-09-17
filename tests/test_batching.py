"""Batching: the 32 prompts through the engine, staggered (one admitted per step) and all at
once, finishing at different lengths. Every request's output equals its batch-1 run,
which equals HF greedy. The pool must be fully returned at the end.

One allowance: GEMM kernels are not batch-invariant (the reduction order depends on M),
so a step whose top-2 logits tie within bf16 tolerance may legitimately resolve
differently in a batch. A divergence is accepted only at such a step, and the number of
prompts that hit one is reported."""

from __future__ import annotations

import pytest
import torch

from engine.engine import Engine, EngineConfig
from sched.scheduler import SamplingParams
from tests.conftest import MAX_NEW_TOKENS, assert_tokens_match

BLOCK_SIZE = 16
MAX_SEQ = 1024
NUM_BLOCKS = 32 * (MAX_SEQ // BLOCK_SIZE)


@pytest.fixture(scope="module")
def prompts(encoded):
    return [ids.tolist() for ids in encoded]


@pytest.fixture(scope="module")
def batch1(ns, prompts):
    """Eager, one request at a time: (tokens, per-step logits) per prompt -- the reference
    every batched / graphed / speculative run is compared against."""
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=1, cuda_graphs=False, record_logits=True))
    out = []
    for p in prompts:
        r = eng.add(p, SamplingParams(MAX_NEW_TOKENS))
        while eng.sched.has_work:
            eng.step()
        out.append((r.out_tokens, torch.stack(r.step_logits)))
    return out


def test_batch1_matches_hf(batch1, hf_ref, strict):
    ties = [t for i, ((ours, _), (ref, lg)) in enumerate(zip(batch1, hf_ref))
            if (t := assert_tokens_match(ours, ref, lg, strict, f"prompt {i}"))]
    if ties:
        print(f"\n[batching batch1] {len(ties)}/{len(batch1)} prompts diverged from HF on a bf16 tie")


@pytest.mark.parametrize("max_admit", [1, 4, 32], ids=lambda m: f"admit{m}")
def test_batched_matches_batch1(ns, prompts, batch1, max_admit):
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=max_admit, cuda_graphs=False))
    outs = eng.generate(prompts, SamplingParams(MAX_NEW_TOKENS))
    # never strict: GEMMs are not batch-invariant on any backend
    ties = [(i, *t) for i, (ours, (ref, lg)) in enumerate(zip(outs, batch1))
            if (t := assert_tokens_match(ours, ref, lg, False, f"prompt {i} (max_admit={max_admit})"))]
    assert eng.alloc.num_free == eng.usable_blocks
    eng.alloc.check()
    if ties:
        print(f"\n[batching admit={max_admit}] {len(ties)}/{len(prompts)} prompts diverged on a bf16 tie: {ties}")


def test_lengths_differ_and_pool_drains(ns, prompts):
    """Finish order != arrival order exercises free() mid-flight."""
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=1))
    params = [SamplingParams(max_tokens=(i % 5) * 7 + 3) for i in range(len(prompts))]
    reqs = [eng.add(p, s) for p, s in zip(prompts, params)]
    finish_order = []
    while eng.sched.has_work:
        finish_order += [r.id for r in eng.step()]
        eng.alloc.check()
    assert all(len(r.out_tokens) <= s.max_tokens for r, s in zip(reqs, params))
    assert finish_order != sorted(finish_order)
    assert eng.alloc.num_free == eng.usable_blocks
