"""G3: the 32 prompts through the engine, staggered (one admitted per step) and all at
once, finishing at different lengths. Every request's output equals its batch-1 run,
which equals HF greedy. The pool must be fully returned at the end.

One allowance: GEMM kernels are not batch-invariant (the reduction order depends on M),
so a step whose top-2 logits tie within bf16 tolerance may legitimately resolve
differently in a batch. A divergence is accepted only at such a step, and the number of
prompts that hit one is reported."""

from __future__ import annotations

import pytest

from engine.engine import Engine, EngineConfig
from sched.scheduler import SamplingParams
from tests.conftest import MAX_NEW_TOKENS, logit_tol

BLOCK_SIZE = 16
MAX_SEQ = 1024
NUM_BLOCKS = 32 * (MAX_SEQ // BLOCK_SIZE)


@pytest.fixture(scope="module")
def prompts(encoded):
    return [ids.tolist() for ids in encoded]


@pytest.fixture(scope="module")
def batch1(ns, prompts):
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=1))
    return [eng.generate([p], SamplingParams(MAX_NEW_TOKENS))[0] for p in prompts]


def test_batch1_matches_hf(batch1, hf_ref):
    for i, (ours, (ref, _)) in enumerate(zip(batch1, hf_ref)):
        assert ours == ref, f"prompt {i}"


def _first_divergence(ours: list[int], ref: list[int]) -> int | None:
    for k, (a, b) in enumerate(zip(ours, ref)):
        if a != b:
            return k
    return None if len(ours) == len(ref) else min(len(ours), len(ref))


@pytest.mark.parametrize("max_admit", [1, 4, 32], ids=lambda m: f"admit{m}")
def test_batched_matches_batch1(ns, prompts, batch1, hf_ref, max_admit):
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=max_admit))
    outs = eng.generate(prompts, SamplingParams(MAX_NEW_TOKENS))
    ties = []
    for i, (ours, ref) in enumerate(zip(outs, batch1)):
        k = _first_divergence(ours, ref)
        if k is None:
            continue
        top2 = hf_ref[i][1][k].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        assert gap <= logit_tol(top2[0]).item(), (
            f"prompt {i} (max_admit={max_admit}) diverged at step {k} with a clear top-2 gap {gap:.4f}\n"
            f"ours={ours}\nref ={ref}"
        )
        ties.append((i, k, gap))
    assert eng.alloc.num_free == NUM_BLOCKS
    eng.alloc.check()
    if ties:
        print(f"\n[G3 admit={max_admit}] {len(ties)}/{len(prompts)} prompts diverged on a bf16 tie: {ties}")


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
    assert eng.alloc.num_free == NUM_BLOCKS
