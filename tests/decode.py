"""Reference single-request greedy loop over the paged cache. The engine replaces this."""

from __future__ import annotations

import torch

from kv.allocator import BlockAllocator
from kv.cache import AttnMeta, PagedKVCache
from model.llama import LlamaForCausalLM


def greedy(
    model: LlamaForCausalLM,
    kv: PagedKVCache,
    alloc: BlockAllocator,
    req_id: int,
    ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], torch.Tensor]:
    """ids: [S]. Returns generated tokens and per-step next-token logits [T, V]."""
    device = ids.device
    eos = set(model.config.eos_token_ids)
    tokens: list[int] = []
    step_logits: list[torch.Tensor] = []

    table = alloc.alloc(req_id, ids.shape[0])
    meta = AttnMeta.build([table], [ids.shape[0]], kv.block_size, device)
    logits, _ = model(ids, kv, meta, logits_idx=meta.last_token_idx)
    for _ in range(max_new_tokens):
        step_logits.append(logits[0])
        nxt = int(logits[0].argmax())
        tokens.append(nxt)
        if nxt in eos:
            break
        alloc.append(req_id, 1)
        meta = AttnMeta.build([table], [1], kv.block_size, device)
        logits, _ = model(torch.tensor([nxt], device=device), kv, meta)
    alloc.free(req_id)
    return tokens, torch.stack(step_logits)
