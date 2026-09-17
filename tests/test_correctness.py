"""Greedy decode of 32 prompts x 128 tokens matches HF transformers greedy
token-for-token, and logits agree within bf16 tolerance at every position.

The "plain" case runs on a fresh pool with a large block size; "fragmented" uses block_size=16
on a pool fragmented by alloc/free churn first.

"""

from __future__ import annotations

import random

import pytest
import torch

from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from tests.conftest import DEVICE, DTYPE, MAX_NEW_TOKENS, assert_logits_close, assert_tokens_match
from tests.decode import greedy
from tests.prompts import G1_PROMPTS

MAX_SEQ = 1024

# case -> (block_size, fragment the pool first)
CASES = {"plain": (256, False), "fragmented": (16, True)}


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def paged(request, ns):
    block_size, fragment = CASES[request.param]
    # enough for 4 max-length requests plus churn headroom
    num_blocks = 4 * -(-MAX_SEQ // block_size) + 64
    kv = PagedKVCache(ns.config, num_blocks, block_size, DEVICE, DTYPE)
    alloc = BlockAllocator(num_blocks, block_size)
    if fragment:
        rng = random.Random(0)
        live = []
        for i in range(200):
            if live and rng.random() < 0.5:
                alloc.free(live.pop(rng.randrange(len(live))))
            elif alloc.num_free > 64:
                alloc.alloc(-1 - i, rng.randint(1, 4 * block_size))
                live.append(-1 - i)
        for r in live[::2]:
            alloc.free(r)
        alloc.check()
        probe = alloc.alloc(-10**6, MAX_SEQ).blocks  # the test prompts must land on scattered blocks
        assert any(b != a + 1 for a, b in zip(probe, probe[1:])), probe
        alloc.free(-10**6)
    return kv, alloc


@pytest.mark.parametrize("i", range(len(G1_PROMPTS)), ids=lambda i: f"p{i:02d}")
def test_greedy_matches_hf(ns, strict, paged, encoded, hf_ref, i):
    kv, alloc = paged
    ref_tokens, ref_logits = hf_ref[i]
    our_tokens, our_logits = greedy(ns, kv, alloc, i, encoded[i], MAX_NEW_TOKENS)
    alloc.check()

    tie = assert_tokens_match(our_tokens, ref_tokens, ref_logits, strict, f"prompt {i}")
    n = tie[0] if tie else min(len(our_logits), len(ref_logits))
    assert_logits_close(our_logits[:n], ref_logits[:n], f"prompt {i} decode-step logits", strict)
    if tie:
        print(f"\n[correctness] prompt {i}: diverged from HF at step {tie[0]} on a bf16 tie (gap {tie[1]:.4f})")


@pytest.mark.parametrize("i", range(len(G1_PROMPTS)), ids=lambda i: f"p{i:02d}")
def test_prefill_logits_every_position(hf, ns, strict, paged, encoded, hf_ref, i):
    """Teacher-forced forward over prompt + HF's continuation: logits at every position."""
    _, hf_model = hf
    kv, alloc = paged
    full = torch.cat([encoded[i], torch.tensor(hf_ref[i][0], device=DEVICE)])

    with torch.no_grad():
        ref = hf_model(full[None], attention_mask=torch.ones_like(full[None])).logits[0]
    table = alloc.alloc(1000 + i, full.shape[0])
    meta = AttnMeta.build([table], [full.shape[0]], kv.block_size, DEVICE)
    ours, _ = ns(full, kv, meta)
    alloc.free(1000 + i)
    assert_logits_close(ours, ref, f"prompt {i} prefill logits", strict)
