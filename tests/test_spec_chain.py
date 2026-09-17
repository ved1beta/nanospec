"""EAGLE-3 chain (topk=1, depth=5). Output == the no-spec engine output (greedy
speculation is lossless; tie-aware for kernel noise), KV pool fully returned, and the
mean accepted length per step is reported for comparison with SGLang.

Locally (tiny model, no trained head) the target itself poses as the drafter, with a
wrong token injected at every third position, so acceptance varies over 0..D and every
verify / rollback / extend path runs. On the H100 set
NANOSPEC_EAGLE=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B for the real head."""

from __future__ import annotations

import os

import pytest
import torch

from engine.engine import Engine, EngineConfig
from model.loader import load_eagle3
from sched.scheduler import SamplingParams
from tests.conftest import BACKEND, MAX_NEW_TOKENS, assert_tokens_match
from tests.test_batching import BLOCK_SIZE, NUM_BLOCKS

EAGLE = os.environ.get("NANOSPEC_EAGLE")
DEPTH = 5


class SelfDrafter:
    """The target model as its own drafter (full-depth draft cache), with the argmax
    forced wrong wherever position % 3 == 2. Same interface as Eagle3Head."""

    class _Cfg:
        def __init__(self, target):
            self.target = target

        def rope_config(self):
            return self.target.config

    def __init__(self, target):
        self.target = target
        self.config = self._Cfg(target)

    def __call__(self, ids, hidden, kv, meta):
        logits, _ = self.target(ids, kv, meta)
        wrong = (meta.positions % 3 == 2).nonzero().flatten()
        if len(wrong):
            top = logits[wrong].argmax(-1)
            logits[wrong] = -1e4
            logits[wrong, (top + 1) % logits.shape[-1]] = 1e4
        return logits, torch.zeros(len(ids), 1, device=ids.device, dtype=ids.dtype)

    def to_target_ids(self, x):
        return x


@pytest.fixture(scope="module")
def drafter(ns):
    return load_eagle3(EAGLE, ns, backend=BACKEND) if EAGLE else SelfDrafter(ns)


@pytest.fixture(scope="module")
def prompts(encoded):
    return [ids.tolist() for ids in encoded]


from tests.test_batching import batch1 as plain  # noqa: F401  (tokens, logits) per prompt


@pytest.mark.parametrize("max_admit", [1, 32], ids=lambda m: f"admit{m}")
def test_chain_is_lossless(ns, drafter, prompts, plain, max_admit):
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=max_admit, cuda_graphs=False, spec_depth=DEPTH), drafter)
    reqs = [eng.add(p, SamplingParams(MAX_NEW_TOKENS)) for p in prompts]
    while eng.sched.has_work:
        eng.step()
        eng.alloc.check()
    ties = [(i, *t) for i, r in enumerate(reqs)
            if (t := assert_tokens_match(r.out_tokens, plain[i][0], plain[i][1], False, f"prompt {i} spec vs plain"))]
    assert eng.alloc.num_free == eng.usable_blocks
    steps = sum(len(r.accepted) for r in reqs)
    tokens = sum(len(r.out_tokens) for r in reqs)
    acc = sum(sum(r.accepted) for r in reqs) / max(steps, 1)
    print(f"\n[spec chain admit={max_admit}] mean accepted drafts/step {acc:.2f} (+1 bonus = {acc + 1:.2f} tokens/step), "
          f"{tokens} tokens in {steps + len(reqs)} target forwards; ties {ties}")


@pytest.mark.skipif(not EAGLE, reason="needs the trained head")
@pytest.mark.parametrize("topk", [1, 4, 6], ids=["chain", "tree4", "tree6"])
def test_acceptance_on_mt_bench(ns, drafter, hf, topk):
    """Mean accepted length per step on chat-templated MT-Bench (the setting SGLang's
    EAGLE-3 numbers are quoted for): chain >= 2.5 tokens/step, tree >= chain."""
    from bench.prompts import chat_prompts

    tok, _ = hf
    prompts = [tok(p, add_special_tokens=False).input_ids for p in chat_prompts(tok)[:20]]
    eng = Engine(ns, EngineConfig(NUM_BLOCKS, BLOCK_SIZE, max_admit=4, cuda_graphs=False, spec_depth=DEPTH, spec_topk=topk), drafter)
    reqs = [eng.add(p, SamplingParams(128)) for p in prompts]
    while eng.sched.has_work:
        eng.step()
    steps = sum(len(r.accepted) for r in reqs)
    tps = sum(sum(r.accepted) for r in reqs) / steps + 1
    print(f"\n[spec MT-Bench {'tree' if topk > 1 else 'chain'} depth={DEPTH} topk={topk}] {tps:.2f} tokens/step over {steps} steps")
    assert tps >= 2.5
