"""G1 / G2: greedy decode of 32 prompts x 128 tokens matches HF transformers greedy
token-for-token, and logits agree within bf16 tolerance at every position.

G1 runs on a fresh pool with a large block size; G2 is the same test with block_size=16
on a pool fragmented by alloc/free churn first.

"""

from __future__ import annotations

import random

import pytest
import torch

from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from tests.conftest import DEVICE, DTYPE, MAX_NEW_TOKENS, logit_tol
from tests.decode import greedy
from tests.prompts import G1_PROMPTS

MAX_SEQ = 1024

# gate -> (block_size, fragment the pool first)
GATES = {"g1": (256, False), "g2": (16, True)}


@pytest.fixture(scope="module", params=list(GATES), ids=list(GATES))
def paged(request, ns):
    block_size, fragment = GATES[request.param]
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


def _assert_logits_close(ours: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    ours, ref = ours.float(), ref.float()
    diff = (ours - ref).abs()
    bad = diff > logit_tol(ref)
    if bad.any():
        pos = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(
            f"{where}: {int(bad.sum())} logits outside bf16 tolerance; worst |diff|={diff.max().item():.4f}, "
            f"first at {pos}: ours={ours[pos].item():.4f} ref={ref[pos].item():.4f}"
        )


@pytest.mark.parametrize("i", range(len(G1_PROMPTS)), ids=lambda i: f"p{i:02d}")
def test_greedy_matches_hf(ns, paged, encoded, hf_ref, i):
    kv, alloc = paged
    ref_tokens, ref_logits = hf_ref[i]
    our_tokens, our_logits = greedy(ns, kv, alloc, i, encoded[i], MAX_NEW_TOKENS)
    alloc.check()

    if our_tokens != ref_tokens:
        k = next((k for k, (a, b) in enumerate(zip(our_tokens, ref_tokens)) if a != b), min(len(our_tokens), len(ref_tokens)))
        raise AssertionError(f"prompt {i}: diverged at step {k}\nours={our_tokens}\nref ={ref_tokens}")
    _assert_logits_close(our_logits, ref_logits, f"prompt {i} decode-step logits")


@pytest.mark.parametrize("i", range(len(G1_PROMPTS)), ids=lambda i: f"p{i:02d}")
def test_prefill_logits_every_position(hf, ns, paged, encoded, hf_ref, i):
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
    _assert_logits_close(ours, ref, f"prompt {i} prefill logits")
